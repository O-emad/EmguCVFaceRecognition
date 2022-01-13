using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using Size = System.Drawing.Size;
using Rect = System.Drawing.Rectangle;
using Emgu.CV.Structure;
using System.Runtime.InteropServices;
using Emgu.CV;
using Emgu.CV.CvEnum;
using System;
namespace EmguCVFaceRecognition
{
    public static class ObjectDetection
    {
        public static Rect[] DetectObjectsCustom(Mat img, CascadeClassifier classifier, int scaledWidth, float searchScaleFactor, int minNeighbors, Size minFeatureSize)
        {
            Mat grayImage = new Mat();
            if (img.NumberOfChannels == 3)
            {
                CvInvoke.CvtColor(img, grayImage, ColorConversion.Bgr2Gray);
            }
            else if (img.NumberOfChannels == 4)
            {
                CvInvoke.CvtColor(img, grayImage, ColorConversion.Bgra2Gray);
            }
            else
            {
                grayImage = img;
            }

            Mat inputImage = new Mat();
            float scale = img.Cols / (float)scaledWidth;
            if (img.Cols > scaledWidth)
            {
                // Shrink the image while keeping the same aspect ratio.
                int scaledHeight = (int)Math.Round(img.Rows / scale);
                CvInvoke.Resize(grayImage, inputImage, new Size(scaledWidth, scaledHeight));
            }
            else
            {
                // Access the input image directly, since it is already small.
                inputImage = grayImage;
            }

            Mat equalizedImage = new Mat();
            CvInvoke.EqualizeHist(inputImage, equalizedImage);


            var objects = classifier.DetectMultiScale(equalizedImage, searchScaleFactor, minNeighbors, minFeatureSize);

            // Enlarge the results if the image was temporarily shrunk before detection.
            if (img.Cols > scaledWidth)
            {
                for (int i = 0; i < objects.Length; i++)
                {
                    objects[i].X = (int)Math.Round(objects[i].X * scale);
                    objects[i].Y = (int)Math.Round(objects[i].Y * scale);
                    objects[i].Width = (int)Math.Round(objects[i].Width * scale);
                    objects[i].Height = (int)Math.Round(objects[i].Height * scale);
                }
            }

            // Make sure the object is completely within the image, in case it was on a border.
            for (int i = 0; i < objects.Length; i++)
            {
                if (objects[i].X < 0)
                    objects[i].X = 0;
                if (objects[i].Y < 0)
                    objects[i].Y = 0;
                if (objects[i].X + objects[i].Width > img.Cols)
                    objects[i].X = img.Cols - objects[i].Width;
                if (objects[i].Y + objects[i].Height > img.Rows)
                    objects[i].Y = img.Rows - objects[i].Height;
            }

            return objects;
        }

        public static Rect DetectLargestObject(Mat img, CascadeClassifier classifier, int scaledWidth)
        {
            Size minFeatureSize = new Size(20, 20);
            // How detailed should the search be. Must be larger than 1.0.
            float searchScaleFactor = 1.1f;
            // How much the detections should be filtered out. This should depend on how bad false detections are to your system.
            // minNeighbors=2 means lots of good+bad detections, and minNeighbors=6 means only good detections are given but some are missed.
            int minNeighbors = 4;

            var result = DetectObjectsCustom(img, classifier, scaledWidth, searchScaleFactor, minNeighbors, minFeatureSize);

            if (result.Length > 0) return result[0];
            else return new Rect(-1, -1, -1, -1);

        }
    }
}
