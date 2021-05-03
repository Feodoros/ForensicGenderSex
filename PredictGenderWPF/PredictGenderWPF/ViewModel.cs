using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using FaceDetector;
using FaceDetector.CenterFace;

namespace PredictGenderWPF
{
    class ViewModel : INotifyPropertyChanged
    {
        private CenterFace _centerFace;
        private GenderPredictor _genderPredictor;
        private readonly List<PrepairedFace> Data = new List<PrepairedFace>();

        #region Actions/Funcs

        public event PropertyChangedEventHandler PropertyChanged;

        public Action<BitmapSource> SetImage;
        public Action<BitmapSource> SetSmallImage;
        public Action<List<PrepairedFace>> SetListBoxFaces;
        public Action<string> LogMessage;
        public Action<string> SetSelectedLbItem;
        public Action<string> SetGenderLabel;

        public Func<BitmapSource> GetOriginalImage;
        public Func<BitmapSource> GetCurrentImage;
        public Func<string> GetSavingPath;
        public Func<string> GetOriginalImagePath;
        public Func<double> GetScaleFactor;

        #endregion

        #region Commands

        private ICommand _analyzeCommand;
        private ICommand _saveCommand;
        private ICommand _selectFaceCommand;
        private ICommand _clickFaceCommand;

        public ICommand AnalyzeCommand
        {
            get
            {
                if (_analyzeCommand == null)
                {
                    _analyzeCommand = new RelayCommand(o => Analyze(), o => true);
                }
                return _analyzeCommand;
            }
        }

        public ICommand SelectFaceCommand
        {
            get
            {
                if (_selectFaceCommand == null)
                {
                    _selectFaceCommand = new RelayCommand(SelectFace, o => true);
                }
                return _selectFaceCommand;
            }
        }

        public ICommand ClickFaceCommand
        {
            get
            {
                if (_clickFaceCommand == null)
                {
                    _clickFaceCommand = new RelayCommand(o => ClickOnImage(), o => true);
                }
                return _clickFaceCommand;
            }
        }

        public ICommand SaveCommand
        {
            get
            {
                if (_saveCommand == null)
                {
                    _saveCommand = new RelayCommand(o => SaveImage(), o => true);
                }
                return _saveCommand;
            }
        }

        #endregion

        #region Fields        

        private bool _isPictureSet;
        private double _panelX;
        private double _panelY;

        #endregion

        #region Properties

        public bool IsControlsEnabled
        {
            get
            {
                return _isPictureSet;
            }
            private set
            {
                _isPictureSet = value;
                RaisePropertyChanged("IsControlsEnabled");
            }
        }

        public double PanelX
        {
            get { return _panelX; }
            set
            {
                if (value.Equals(_panelX)) return;
                _panelX = value;
                RaisePropertyChanged("MouseX");
            }
        }

        public double PanelY
        {
            get { return _panelY; }
            set
            {
                if (value.Equals(_panelY)) return;
                _panelY = value;
                RaisePropertyChanged("MouseY");
            }
        }

        #endregion

        #region GenderPrediction

        private async void Analyze()
        {
            UpdateControlsEnableState(false);

            Data.Clear();
            PrepareTempDirectory();

            if (_centerFace == null)
            {
                _centerFace = new CenterFace();
            }

            if (_genderPredictor == null)
            {
                _genderPredictor = new GenderPredictor();
            }

            await Task.Run(() =>
            {
                RunGenderEstimation();
            });

            UpdateControlsEnableState(true);
            SetListBoxFaces?.Invoke(Data);
            SetSelectedLbItem?.Invoke(Data.FirstOrDefault().Name);
            DrawRectsOnImage();
        }

        private void RunGenderEstimation()
        {
            string currentImagePath = GetOriginalImagePath?.Invoke();
            if (string.IsNullOrEmpty(currentImagePath))
            {
                return;
            }

            DetectFaces(currentImagePath);

            CropFaces();

            PredictGender();

            Logger($"Gender estimation finished");
        }

        private void DetectFaces(string currentImagePath)
        {
            try
            {
                List<Face> faces = _centerFace.DetectFaces(currentImagePath);
                if (faces.Count == 0)
                {
                    Logger($"No face detected");
                    return;
                }
                Logger($"Detected {faces.Count} faces");

                foreach (Face face in faces)
                {
                    Data.Add(new PrepairedFace
                    {
                        X1 = face.X1,
                        X2 = face.X2,
                        Y1 = face.Y1,
                        Y2 = face.Y2
                    });
                }
            }
            catch (Exception ex)
            {
                Logger("Catch exception: " + ex.Message);
                return;
            }
        }

        private void CropFaces()
        {
            for (int i = 0; i < Data.Count; i++)
            {
                PrepairedFace prepairedFace = Data[i];
                string cropfaceName = $"Face_{i}";
                string cropFacePath = Path.Combine(GetTempDirectoryPath(), $"{cropfaceName}.jpg");

                prepairedFace.Name = cropfaceName;
                prepairedFace.Path = cropFacePath;

                CroppedBitmap cropFace = CropFaceFromBitmap(prepairedFace);

                BitmapEncoder encoder = new JpegBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(cropFace));
                using (FileStream fileStream = new FileStream(cropFacePath, FileMode.Create))
                {
                    encoder.Save(fileStream);
                }
            }
        }

        private void PredictGender()
        {
            foreach (PrepairedFace prepairedFace in Data)
            {
                string gender = "Unknown";
                try
                {
                    gender = _genderPredictor.Predict(prepairedFace.Path);
                }
                catch (Exception ex)
                {
                    Logger("Catch exception: " + ex.Message);
                }
                prepairedFace.Gender = gender;
            }
        }

        #endregion

        #region Helpers

        private CroppedBitmap CropFaceFromBitmap(PrepairedFace prepairedFace)
        {
            string pathToImage = GetOriginalImagePath?.Invoke();
            BitmapImage bitmapImage = new BitmapImage();
            bitmapImage.BeginInit();
            bitmapImage.UriSource = new Uri(pathToImage, UriKind.Absolute);
            bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
            bitmapImage.EndInit();

            double width = bitmapImage.Width;
            double height = bitmapImage.Height;

            int x1 = (int)Math.Max(prepairedFace.X1 - prepairedFace.X2 * 0.25, 0);
            int y1 = (int)Math.Max(prepairedFace.Y1 - prepairedFace.Y2 * 0.25, 0);
            int x2 = (int)Math.Min(prepairedFace.X2 * 1.25, width);
            int y2 = (int)Math.Min(prepairedFace.Y2 * 1.25, height);

            CroppedBitmap cb = new CroppedBitmap(bitmapImage, new Int32Rect(x1, y1, x2, y2));
            return cb;
        }

        private void PrepareTempDirectory()
        {
            string tempDir = GetTempDirectoryPath();
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, true);
            }
            Directory.CreateDirectory(tempDir);
            Logger("Temp directory: " + tempDir);
        }

        private string GetTempDirectoryPath()
        {
            string projectDirectory = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;
            return Path.Combine(projectDirectory, "temp");
        }

        private void ChangeFaceSelection()
        {
            foreach (PrepairedFace face in Data)
            {
                if (face.Selected)
                {
                    string pathToFace = face.Path;
                    BitmapImage bitmapImage = new BitmapImage();
                    bitmapImage.BeginInit();
                    bitmapImage.UriSource = new Uri(pathToFace, UriKind.Absolute);
                    bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                    bitmapImage.EndInit();

                    DrawingVisual dv = new DrawingVisual();
                    RenderTargetBitmap rtb = new RenderTargetBitmap(bitmapImage.PixelWidth, bitmapImage.PixelHeight, 96, 96, PixelFormats.Pbgra32);

                    using (DrawingContext dc = dv.RenderOpen())
                    {
                        dc.DrawImage(bitmapImage, new Rect(0, 0, bitmapImage.PixelWidth, bitmapImage.PixelHeight));
                    }

                    rtb.Render(dv);

                    SetSmallImage?.Invoke(rtb);
                    SetGenderLabel?.Invoke(face.Gender);
                    break;
                }
            }
            DrawRectsOnImage();
        }

        public void UpdateControlsEnableState(bool isEnabled)
        {
            IsControlsEnabled = isEnabled;
        }

        private void DrawRectsOnImage()
        {
            BitmapSource originalImage = GetOriginalImage?.Invoke().Clone();

            double size = originalImage.PixelWidth * originalImage.PixelHeight;
            double scaleFactor = GetScaleFactor.Invoke();
            double rectThickness = size / (750 * 750);
            double fontSize = originalImage.PixelHeight / 30;

            Pen redPen = new Pen(Brushes.Red, rectThickness);
            Pen greenPen = new Pen(Brushes.LawnGreen, rectThickness);

            DrawingVisual dv = new DrawingVisual();
            RenderTargetBitmap rtb = new RenderTargetBitmap(originalImage.PixelWidth, originalImage.PixelHeight, 96, 96, PixelFormats.Pbgra32);

            using (DrawingContext dc = dv.RenderOpen())
            {
                dc.DrawImage(originalImage, new Rect(0, 0, originalImage.PixelWidth, originalImage.PixelHeight));

                foreach (PrepairedFace face in Data)
                {
                    Pen pen = greenPen;
                    if (face.Selected)
                    {
                        pen = redPen;
                    }

                    dc.DrawRectangle(null, pen, new Rect(face.X1, face.Y1, face.X2, face.Y2));

                    FormattedText text = new FormattedText(face.Gender,
                        CultureInfo.GetCultureInfo("en-us"),
                        FlowDirection.LeftToRight,
                        new Typeface("Tahoma"),
                        fontSize,
                        Brushes.White,
                        96);

                    try
                    {
                        dc.DrawText(text, new Point(face.X1 - rectThickness, face.Y1 - fontSize - rectThickness));
                    }
                    catch
                    {
                        dc.DrawText(text, new Point(face.X1, face.Y1));
                    }
                }
            }

            rtb.Render(dv);

            SetImage?.Invoke(rtb);
            Logger("Drawing rectangles finished");
        }

        public void ChangePicture()
        {
            Data.Clear();
            Logger("Picture was changed");
        }

        private void Logger(string message)
        {
            Application.Current.Dispatcher.Invoke(() => LogMessage?.Invoke(message));
        }

        #endregion

        #region UserControls

        private void ClickOnImage()
        {
            double factor = GetScaleFactor.Invoke();
            double newX = PanelX * factor;
            double newY = PanelY * factor;

            foreach (PrepairedFace face in Data)
            {
                face.Selected = false;

                // Check required face
                if (newX > face.X1 && newX < (face.X1 + face.X2) &&
                    newY > face.Y1 && newY < (face.Y1 + face.Y2))
                {
                    face.Selected = true;
                    ChangeFaceSelection();
                    SetSelectedLbItem?.Invoke(face.Name);
                    Logger("Face touched, selected new face");
                }
            }
        }

        private void SelectFace(object parameter)
        {
            if (parameter == null)
            {
                return;
            }

            ListBox lb = (ListBox)parameter;
            ListBoxItem selectedItem = (ListBoxItem)lb.SelectedItem;
            if (lb.Items.Count == 0 || lb.Items == null || selectedItem == null)
            {
                return;
            }

            string content = (string)selectedItem.Content;
            string faceName = content.Split(',').FirstOrDefault();
            foreach (PrepairedFace face in Data)
            {
                face.Selected = faceName.Equals(face.Name);
            }
            ChangeFaceSelection();
        }

        private void SaveImage()
        {
            string pathToSave = GetSavingPath?.Invoke();
            if (string.IsNullOrEmpty(pathToSave))
            {
                return;
            }

            BitmapSource bitmapSource = GetCurrentImage?.Invoke();
            BitmapEncoder encoder = new JpegBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(bitmapSource));

            using (FileStream fileStream = new FileStream(pathToSave, FileMode.Create))
            {
                encoder.Save(fileStream);
                Logger("Image saved in " + pathToSave);
            }
        }

        #endregion

        private void RaisePropertyChanged(string PropertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(PropertyName));
        }
    }
}
