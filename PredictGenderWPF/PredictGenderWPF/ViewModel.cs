using FaceDetector;
using FaceDetector.CenterFace;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Threading;

namespace PredictGenderWPF
{
    class ViewModel : INotifyPropertyChanged
    {
        private List<PrepairedFace> _prepairedFaces = new List<PrepairedFace>();
        private CenterFace _centerFace;
        private GenderPredictor _genderPredictor;
        public event PropertyChangedEventHandler PropertyChanged;

        public Action<Bitmap> SetImage;
        public Action<Bitmap> SetSmallImage;
        public Action<List<PrepairedFace>> SetListBoxFaces;
        public Action<string> LogMessage;
        public Action<string> SetSelectedLbItem;
        public Func<Bitmap> GetOriginalImage;
        public Func<Bitmap> GetCurrentImage;
        public Func<string> GetSavingPath;
        public Func<string> GetCurrentImagePath;
        public Func<float> GetScaleFactor;

        #region Fields        
        private ICommand _analyzeCommand;
        private ICommand _saveCommand;
        private ICommand _selectFaceCommand;
        private ICommand _clickFaceCommand;
        private bool _isPictureSet;
        #endregion

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
        private double _panelX;
        private double _panelY;
        public double PanelX
        {
            get { return _panelX; }
            set
            {
                if (value.Equals(_panelX)) return;
                _panelX = value;
                RaisePropertyChanged("");
            }
        }

        public double PanelY
        {
            get { return _panelY; }
            set
            {
                if (value.Equals(_panelY)) return;
                _panelY = value;
                RaisePropertyChanged("");
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

        private void SaveImage()
        {
            string pathToSave = GetSavingPath?.Invoke();
            if (string.IsNullOrEmpty(pathToSave))
            {
                return;
            }

            using (Bitmap currentImage = new Bitmap(GetCurrentImage?.Invoke()))
            {
                currentImage.Save(pathToSave);
                Logger("Image saved in " + pathToSave);
            }
        }

        private void ClickOnImage()
        {
            float factor = GetScaleFactor.Invoke();
            double newX = PanelX * factor;
            double newY = PanelY * factor;

            foreach (PrepairedFace face in _prepairedFaces)
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
            foreach (PrepairedFace face in _prepairedFaces)
            {
                face.Selected = faceName.Equals(face.Name);
            }
            ChangeFaceSelection();
        }

        private void ChangeFaceSelection()
        {
            foreach (PrepairedFace face in _prepairedFaces)
            {
                if (face.Selected)
                {
                    string pathToFace = face.Path;
                    using (Bitmap smallFace = new Bitmap(pathToFace))
                    using (Graphics g = Graphics.FromImage(smallFace))
                    {
                        g.DrawString(face.Gender, new Font("Tahoma", face.X2 / 5), Brushes.White, 0, 0);
                        SetSmallImage?.Invoke(smallFace);
                    }
                }
            }
            DrawRectsOnImage();
        }

        public void UpdateControlsEnableState(bool isEnabled)
        {
            IsControlsEnabled = isEnabled;
        }

        public string FaceByClick(double x, double y)
        {
            string facePath = "";

            foreach (PrepairedFace face in _prepairedFaces)
            {
                // Check required face
                if (x > face.X1 && x < (face.X1 + face.X2) &&
                    y > face.Y1 && y < (face.Y1 + face.Y2))
                {
                    return face.Path;
                }
            }

            return facePath;
        }

        private async void Analyze()
        {
            UpdateControlsEnableState(false);
            _prepairedFaces.Clear();

            await Task.Run(() =>
            {
                Run();
            });

            UpdateControlsEnableState(true);
            SetListBoxFaces?.Invoke(_prepairedFaces);
            DrawRectsOnImage();
        }

        private void Run()
        {
            PrepareTempDirectory();
            string currentImagePath = GetCurrentImagePath?.Invoke();
            if (string.IsNullOrEmpty(currentImagePath))
            {
                return;
            }

            if (_centerFace == null)
            {
                _centerFace = new CenterFace();
            }

            if (_genderPredictor == null)
            {
                _genderPredictor = new GenderPredictor();
            }

            // Detect faces
            List<Face> faces = DetectFaces();
            if (faces.Count == 0)
            {
                Logger($"No face detected");
                return;
            }
            Logger($"Detected {faces.Count} faces");

            PredictGender(faces, currentImagePath);
            Logger($"Gender estimation finished");
        }

        private List<Face> DetectFaces()
        {
            List<Face> faces = new List<Face>();
            try
            {
                faces = _centerFace.DetectFaces(GetCurrentImagePath?.Invoke());
            }
            catch (Exception ex)
            {
                Logger("Catch exception: " + ex.Message);
            }
            return faces;
        }

        /// <summary>
        /// Predict gender for each face and make a List type of <see cref="PrepairedFace"/>
        /// </summary>
        /// <param name="faces">Detected faces</param>
        /// <param name="currentImagePath">Path to original image</param>
        private void PredictGender(List<Face> faces, string currentImagePath)
        {
            int i = 0;
            foreach (Face face in faces)
            {
                string cropFacePath = CropFace(face, currentImagePath, i.ToString());
                string gender = "Unknown";
                try
                {
                    gender = _genderPredictor.Predict(cropFacePath);
                }
                catch (Exception ex)
                {
                    Logger("Catch exception: " + ex.Message);
                }
                _prepairedFaces.Add(new PrepairedFace
                {
                    Path = cropFacePath,
                    Name = $"Face_{i}",
                    Gender = gender,
                    X1 = face.X1,
                    X2 = face.X2,
                    Y1 = face.Y1,
                    Y2 = face.Y2
                });

                i++;
            }
        }

        private void DrawRectsOnImage()
        {
            using (Bitmap bitmap = new Bitmap(GetOriginalImage?.Invoke()))
            {
                Pen redPen = new Pen(Color.FromArgb(255, 0, 255), 3);
                Pen greenPen = new Pen(Color.FromArgb(0, 255, 0), 3);

                foreach (PrepairedFace face in _prepairedFaces)
                {
                    Pen pen = greenPen;
                    if (face.Selected)
                    {
                        pen = redPen;
                    }

                    using (Graphics g = Graphics.FromImage(bitmap))
                    {
                        g.DrawRectangle(pen, face.X1, face.Y1, face.X2, face.Y2);
                        try
                        {
                            g.DrawString(face.Gender, new Font("Tahoma", 20), Brushes.White, face.X1 - 5, face.Y1 - 25);
                        }
                        catch
                        {
                            g.DrawString(face.Gender, new Font("Tahoma", 20), Brushes.White, face.X1, face.Y1);
                        }
                    }
                }
                Application.Current.Dispatcher.Invoke(() => SetImage?.Invoke(bitmap));
            }
            Logger("Drawing rectangles finished");
        }

        private string CropFace(Face face, string imagePath, string name)
        {
            Bitmap src = System.Drawing.Image.FromFile(imagePath) as Bitmap;
            int width = src.Width;
            int height = src.Height;

            int x1 = (int)Math.Max(face.X1 - face.X2 * 0.3, 0);
            int y1 = (int)Math.Max(face.Y1 - face.Y2 * 0.3, 0);
            int x2 = (int)Math.Min(face.X2 * 1.3, width);
            int y2 = (int)Math.Min(face.Y2 * 1.3, height);

            Rectangle cropRect = new Rectangle(x1, y1, x2, y2);
            Bitmap target = new Bitmap(cropRect.Width, cropRect.Height);

            using (Graphics g = Graphics.FromImage(target))
            {
                g.DrawImage(src, new Rectangle(0, 0, target.Width, target.Height),
                                 cropRect,
                                 GraphicsUnit.Pixel);
            }
            string cropFacePath = Path.Combine(GetTempDirectoryPath(), $"{name}.jpg");
            target.Save(cropFacePath);
            return cropFacePath;
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

        /// <summary>
        /// User have chosen new pic
        /// </summary>
        public void ChangePicture()
        {
            _prepairedFaces.Clear();
            Logger("Picture was changed");
        }

        private void RaisePropertyChanged(string PropertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(PropertyName));
        }

        private void Logger(string message)
        {
            Application.Current.Dispatcher.Invoke(() => LogMessage?.Invoke(message));
        }
    }
}
