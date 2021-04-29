using Microsoft.Win32;
using Microsoft.WindowsAPICodePack.Dialogs;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace PredictGenderWPF
{
    public partial class MainWindow : Window
    {
        private readonly Action<bool> _updateControlsEnableState;
        private readonly Action<object> _selectFace;
        public delegate void ControlHandler();
        private event ControlHandler ChangePictureEvent;
        private Func<double, double, string> GetFaceByClick;

        private Bitmap OriginalImage { get; set; }
        private Bitmap CurrentImage { get; set; }

        private string CurrentImagePath { get; set; }

        public MainWindow()
        {
            InitializeComponent();
            ViewModel viewModel = new ViewModel
            {
                GetOriginalImage = () => OriginalImage,
                GetCurrentImage = () => CurrentImage,
                GetCurrentImagePath = () => CurrentImagePath,
                GetScaleFactor = GetScaleFactor,
                GetSavingPath = GetImageSavingPath,
                SetImage = (newImage) => { CurrentImage = new Bitmap(newImage); imageBox.Source = BitmapToImageSource(newImage); },
                SetSmallImage = (smallImage) => { imageBoxSmall.Source = BitmapToImageSource(new Bitmap(smallImage)); },
                SetListBoxFaces = (faces) => SetListBoxItems(faces),
                SetSelectedLbItem = (faceName) => SetSelectedLbItem(faceName),
                LogMessage = (message) => Log(message)
            };
            GetFaceByClick = (x, y) => viewModel.FaceByClick(x, y);
            _updateControlsEnableState = viewModel.UpdateControlsEnableState;
            ChangePictureEvent += () => { viewModel.ChangePicture(); };
            DataContext = viewModel;
            Log("Gender estimation using CenterFace detector and EfficientNet-B0 for gender prediction");
        }

        private void SetListBoxItems(List<PrepairedFace> prepairedFaces)
        {
            listBoxFaces.Items.Clear();
            Log("Clear ListBox items");
            foreach (var face in prepairedFaces)
            {
                listBoxFaces.Items.Add(new ListBoxItem() { Content = $"{face.Name}, {face.Gender}" });
            }
            Log($"Add new {prepairedFaces.Count} faces to ListBox");
        }

        private void OpenFile(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "Image files (*.png;*.jpeg;*.jpg)|*.png;*.jpeg;*.jpg|All files (*.*)|*.*"
            };
            if (openFileDialog.ShowDialog() == true)
            {
                ChangePictureEvent?.Invoke();
                CurrentImagePath = openFileDialog.FileName;
                CurrentImage = OriginalImage = new Bitmap(CurrentImagePath);
                imageBox.Source = BitmapToImageSource(OriginalImage);
                Log("Open new image: " + CurrentImagePath);
                _updateControlsEnableState?.Invoke(true);
                listBoxFaces.Items.Clear();
                imageBoxSmall.Source = null;
            }
        }

        private void Log(string message)
        {
            logger.AppendText($"{DateTime.Now}: " + message + '\n');
            logger.ScrollToEnd();
        }

        private string GetImageSavingPath()
        {
            CommonOpenFileDialog dialog = new CommonOpenFileDialog
            {
                IsFolderPicker = true
            };

            if (dialog.ShowDialog() == CommonFileDialogResult.Ok)
            {
                string path = dialog.FileName;
                int count = Directory.GetFiles(path).Where(name => name.Contains("result_image")).Count();
                return $"{path}\\result_image_{count}.jpg";
            }
            return string.Empty;
        }

        private BitmapImage BitmapToImageSource(Bitmap bitmap)
        {
            using (MemoryStream memory = new MemoryStream())
            {
                bitmap.Save(memory, System.Drawing.Imaging.ImageFormat.Bmp);
                memory.Position = 0;
                BitmapImage bitmapimage = new BitmapImage();
                bitmapimage.BeginInit();
                bitmapimage.StreamSource = memory;
                bitmapimage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapimage.EndInit();

                return bitmapimage;
            }
        }

        private float GetScaleFactor()
        {
            float imagewidth = CurrentImage.Width;
            double viewWidth = imageBox.ActualWidth;
            return (float)(imagewidth / viewWidth);
        }

        private void SetSelectedLbItem(string faceName)
        {
            foreach (ListBoxItem item in listBoxFaces.Items)
            {
                string content = (string)item.Content;
                string lbFaceName = content.Split(',').FirstOrDefault();
                if (lbFaceName.Equals(faceName))
                {
                    listBoxFaces.SelectedItem = item;
                    return;
                }
            }
        }

        private void imageBox_MouseDown(object sender, MouseButtonEventArgs e)
        {
            var clickPoint = e.GetPosition(imageBox);
            double x = clickPoint.X;
            double y = clickPoint.Y;

            double imageWidth = imageBox.ActualWidth;
            double imageHeight = imageBox.ActualHeight;
            float width = CurrentImage.Width;
            float height = CurrentImage.Height;
            float sizeRatio = (float)(width / imageWidth);

            double newX = x * sizeRatio;
            double newY = y * sizeRatio;

            string pathToFace = GetFaceByClick?.Invoke(newX, newY);

            if (string.IsNullOrEmpty(pathToFace))
            {
                Log("No face");
                return;
            }
            imageBoxSmall.Source = BitmapToImageSource(new Bitmap(pathToFace));
            listBoxFaces.SelectedItem = listBoxFaces.Items.GetItemAt(0);
            Log("Face touched!" + pathToFace);
            Console.WriteLine();
        }
    }
}
