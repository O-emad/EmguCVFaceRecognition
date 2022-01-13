using System;
using System.Collections.Generic;
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
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Face;
using Emgu.CV.CvEnum;
using System.IO;
using System.Threading;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Emgu.Util;

namespace EmguCVFaceRecognition
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private VideoCapture VideoCapture;
        private Image<Bgr, Byte> CurrentFrame;
        private Mat Frame = new Mat();


        private bool FaceDetectionEnabled;
        private CascadeClassifier FaceCascadeClassifier = new CascadeClassifier("App_Data/haarcascade_frontalface_alt2.xml");
        private CascadeClassifier EyeCascadeClassifier = new CascadeClassifier("App_Data/haarcascade_eye.xml");
        private Image<Gray, byte> FaceResult;
        private List<Mat> TrainedFaces = new List<Mat>();
        private List<int> PersonsLabes = new List<int>();
        List<string> PersonsNames = new List<string>();
        private bool EnableSaveImage;
        private int ImagesToSave = 0;

        private bool IsTrained = false;
        private EigenFaceRecognizer Recognizer;



        /// <summary>
        /// Delete a GDI object
        /// </summary>
        /// <param name="o">The poniter to the GDI object to be deleted</param>
        /// <returns></returns>
        [DllImport("gdi32")]
        private static extern int DeleteObject(IntPtr o);

        /// <summary>
        /// Convert an IImage to a WPF BitmapSource. The result can be used in the Set Property of Image.Source
        /// </summary>
        /// <param name="image">The Emgu CV Image</param>
        /// <returns>The equivalent BitmapSource</returns>
        public static BitmapSource ToBitmapSource(Image<Bgr,byte> image)
        {
            using (System.Drawing.Bitmap source =  image.ToBitmap())
            {
                IntPtr ptr = source.GetHbitmap(); //obtain the Hbitmap

                BitmapSource bs = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                    ptr,
                    IntPtr.Zero,
                    Int32Rect.Empty,
                    System.Windows.Media.Imaging.BitmapSizeOptions.FromEmptyOptions());

                DeleteObject(ptr); //release the HBitmap
                return bs;
            }
        }
        public static BitmapSource ToBitmapSource(Image<Gray, byte> image)
        {
            using (System.Drawing.Bitmap source = image.ToBitmap())
            {
                IntPtr ptr = source.GetHbitmap(); //obtain the Hbitmap

                BitmapSource bs = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                    ptr,
                    IntPtr.Zero,
                    Int32Rect.Empty,
                    System.Windows.Media.Imaging.BitmapSizeOptions.FromEmptyOptions());

                DeleteObject(ptr); //release the HBitmap
                return bs;
            }
        }
        public static BitmapSource ToBitmapSource(Mat image)
        {
            using (System.Drawing.Bitmap source = image.ToBitmap())
            {
                IntPtr ptr = source.GetHbitmap(); //obtain the Hbitmap

                BitmapSource bs = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                    ptr,
                    IntPtr.Zero,
                    Int32Rect.Empty,
                    System.Windows.Media.Imaging.BitmapSizeOptions.FromEmptyOptions());

                DeleteObject(ptr); //release the HBitmap
                return bs;
            }
        }


        public MainWindow()
        {
            InitializeComponent();
            VideoCapture = new VideoCapture();
        }

        private void CaptureButton_Click(object sender, RoutedEventArgs e)
        {
            VideoCapture = new VideoCapture(0);
            VideoCapture.ImageGrabbed += ProcessFrame;
            VideoCapture.Start();
        }

        private void ProcessFrame(object? sender, EventArgs args)
        {
            //video capture
            if (VideoCapture != null)
                VideoCapture.Retrieve(Frame,0);
            CurrentFrame = Frame.ToImage<Bgr, Byte>().Resize(1920, 1080, Inter.Cubic);
            Dispatcher.Invoke(() =>
            {
                
                //Face Detection Enabled
                if (FaceDetectionEnabled)
                {
                    Mat grayImage = new Mat();
                    CvInvoke.CvtColor(CurrentFrame, grayImage, ColorConversion.Bgr2Gray);
                    CvInvoke.EqualizeHist(grayImage, grayImage);

                    Mat processedImage = new Mat();

                    var faces = FaceCascadeClassifier.DetectMultiScale(grayImage, 1.1, 3, new System.Drawing.Size(150,150), System.Drawing.Size.Empty);
                    if (faces.Length > 0)
                    {
                        foreach (var face in faces)
                        {
                            CvInvoke.Rectangle(CurrentFrame, face, new Bgr(System.Drawing.Color.Red).MCvScalar, 2);

                            //add person
                            FaceResult = CurrentFrame.Convert<Gray, Byte>();
                            FaceResult.ROI = face;

                            var processedImage2 = FaceResult.Mat;
                            //System.Drawing.Rectangle? searchedLeftEye = new System.Drawing.Rectangle();
                            //System.Drawing.Rectangle? searchedRightEye = new System.Drawing.Rectangle();
                            //var processedImage2 = FacePreProcessor.GetPreProcessedFace(CurrentFrame.Mat, 200, FaceCascadeClassifier, EyeCascadeClassifier,false
                            //    , new System.Drawing.Rectangle(), new System.Drawing.Point(), new System.Drawing.Point(), ref searchedLeftEye, ref searchedRightEye);
                            try
                            {

                                if (!processedImage2.IsEmpty)
                                    FaceResult = processedImage2.ToImage<Gray, byte>();
                                    CapturedImagePreview.Source = ToBitmapSource(FaceResult);
                                    //CapturedImagePreview.Source =  ToBitmapSource(processedImage2);
                            }
                            catch (Exception) { }
                            //save image
                            if (EnableSaveImage)
                            {
                                var path = Directory.GetCurrentDirectory() + @"\trainedimages";
                                if(!Directory.Exists(path))
                                    Directory.CreateDirectory(path);

                                //save 10 images with a 1 second delay between each one
                                Dispatcher.Invoke(async () => {
                                    for (int i = 0; i < 10; i++)
                                    {
                                        //resize the image then saving it
                                        FaceResult.Resize(200, 200, Inter.Cubic).Save(path + @"\" + PersonName.Text + "_" + DateTime.Now.ToString("dd-mm-yyyy-hh-mm-ss") + ".jpg");
                                        await Task.Delay(1000);
                                    }
                                });
                            }
                            EnableSaveImage = false;

                            if (IsTrained)
                            {
                                var grayFaceResult = FaceResult.Convert<Gray, byte>().Resize(200, 200, Inter.Cubic);
                                CvInvoke.EqualizeHist(grayFaceResult, grayFaceResult);
                                var result = Recognizer.Predict(grayFaceResult);
                                Debug.WriteLine(result.Label + ". " + result.Distance);
                                if (result.Label > 0 && result.Distance < 5000)
                                {
                                    CvInvoke.PutText(CurrentFrame, PersonsNames[result.Label], new System.Drawing.Point(face.X - 2, face.Y - 2),
                                    FontFace.HersheyComplex, 1.0, new Bgr(System.Drawing.Color.Orange).MCvScalar);
                                    CvInvoke.Rectangle(CurrentFrame, face, new Bgr(System.Drawing.Color.Green).MCvScalar, 2);
                                }
                                else
                                {
                                    CvInvoke.PutText(CurrentFrame, "Unknown", new System.Drawing.Point(face.X - 2, face.Y - 2),
                                    FontFace.HersheyComplex, 1.0, new Bgr(System.Drawing.Color.Orange).MCvScalar);
                                    CvInvoke.Rectangle(CurrentFrame, face, new Bgr(System.Drawing.Color.Red).MCvScalar, 2);
                                }
                            }
                        }
                    }
                }
                try
                {
                    VideoStream.Source = ToBitmapSource(CurrentFrame);
                }
                catch (Exception ex) { }
                

            });

        }

        private void DetectFacesButton_Click(object sender, RoutedEventArgs e)
        {
            FaceDetectionEnabled = true;
        }

        private void AddPersonButton_Click(object sender, RoutedEventArgs e)
        {
            SavePerson.IsEnabled = true;
            AddPersonButton.IsEnabled = false;
            EnableSaveImage = true;
        }

        private void SavePerson_Click(object sender, RoutedEventArgs e)
        {
            SavePerson.IsEnabled = false;
            AddPersonButton.IsEnabled = true;
            EnableSaveImage = false;
            ImagesToSave = 0;
        }


        unsafe private bool TrainImagesFromDir()
        {
            int imagesCount = 0;
            double threashold = 5000;
            TrainedFaces.Clear();
            PersonsLabes.Clear();
            PersonsNames.Clear();
            try
            {
                var path = Directory.GetCurrentDirectory() + @"\trainedimages";
                var files = Directory.GetFiles(path,"*.jpg",SearchOption.AllDirectories);

                foreach (var file in files)
                {
                    Image<Gray, byte> trainedImage = new Image<Gray, byte>(file).Resize(200, 200, Inter.Cubic);
                    CvInvoke.EqualizeHist(trainedImage, trainedImage);
                    TrainedFaces.Add(trainedImage.Mat);
                    PersonsLabes.Add(imagesCount);
                    string name = file.Split('\\').Last().Split('_')[0];
                    PersonsNames.Add(name);
                    imagesCount++;
                    Debug.WriteLine(imagesCount + ". " + name);
                }
                Recognizer = new EigenFaceRecognizer(imagesCount,threashold);
                Recognizer.Train(TrainedFaces.ToArray(), PersonsLabes.ToArray());
                IsTrained = true;
                //Debug.WriteLine(imagesCount);
                //Debug.WriteLine(true);
                return true;
            }
            catch (Exception e)
            {

                IsTrained=false;
                MessageBox.Show("Error in train images: " + e.Message);
                return false;
            }



        }

        private void TrainButton_Click(object sender, RoutedEventArgs e)
        {
            TrainImagesFromDir();
        }
    }
}
