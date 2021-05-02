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
        private delegate void ControlHandler();
        private event ControlHandler ChangePictureEvent;
        private readonly Action<bool> _updateControlsEnableState;

        private BitmapImage OriginalImage { get; set; }
        private BitmapSource CurrentImage { get; set; }
        private string OriginalImagePath { get; set; }

        public MainWindow()
        {
            InitializeComponent();
            ViewModel viewModel = new ViewModel
            {
                GetOriginalImage = () => OriginalImage,
                GetCurrentImage = () => CurrentImage,
                GetOriginalImagePath = () => OriginalImagePath,
                GetScaleFactor = GetScaleFactor,
                GetSavingPath = GetImageSavingPath,
                SetImage = (newImage) => { CurrentImage = newImage.Clone(); imageBox.Source = newImage; },
                SetSmallImage = (smallImage) => { imageBoxSmall.Source = smallImage; },
                SetListBoxFaces = (faces) => SetListBoxItems(faces),
                SetSelectedLbItem = (faceName) => SetSelectedLbItem(faceName),
                LogMessage = (message) => Log(message)
            };
            _updateControlsEnableState = viewModel.UpdateControlsEnableState;
            ChangePictureEvent += () => { viewModel.ChangePicture(); };
            DataContext = viewModel;
            Log("Gender estimation using CenterFace detector and EfficientNet-B0 for gender prediction");
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
                listBoxFaces.Items.Clear();
                imageBoxSmall.Source = null;

                OriginalImagePath = openFileDialog.FileName;

                OriginalImage = new BitmapImage();
                OriginalImage.BeginInit();
                OriginalImage.UriSource = new Uri(OriginalImagePath, UriKind.Absolute);
                OriginalImage.CacheOption = BitmapCacheOption.OnLoad;
                OriginalImage.EndInit();

                CurrentImage = OriginalImage.Clone();

                imageBox.Source = OriginalImage;
                Log("Open new image: " + OriginalImagePath);
                _updateControlsEnableState?.Invoke(true);

            }
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

        private double GetScaleFactor()
        {
            double imagewidth = CurrentImage.Width;
            double viewWidth = imageBox.ActualWidth;
            return imagewidth / viewWidth;
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

        private void Log(string message)
        {
            logger.AppendText($"{DateTime.Now}: " + message + '\n');
            logger.ScrollToEnd();
        }
    }
}
