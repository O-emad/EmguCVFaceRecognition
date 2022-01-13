using Emgu.CV;
using Emgu.CV.CvEnum;
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

namespace EmguCVFaceRecognition
{
    public static class FacePreProcessor
    {
        const double DESIRED_LEFT_EYE_X = 0.16;     // Controls how much of the face is visible after preprocessing.
        const double DESIRED_LEFT_EYE_Y = 0.14;
        const double FACE_ELLIPSE_CY = 0.40;
        const double FACE_ELLIPSE_W = 0.50;         // Should be atleast 0.5
        const double FACE_ELLIPSE_H = 0.80;         // Controls how tall the face mask is.

        public static double GetDoubleValue(this Mat mat, int row, int col)
        {
            var value = new double[1];
            Marshal.Copy(mat.DataPointer + (row * mat.Cols + col) * mat.ElementSize, value, 0, 1);
            return value[0];
        }

        public static void SetDoubleValue(this Mat mat, int row, int col, double value)
        {
            var target = new[] { value };
            Marshal.Copy(target, 0, mat.DataPointer + (row * mat.Cols + col) * mat.ElementSize, 1);
        }

        #region RefactorIntoObjectDetectionClass



        #endregion

        public static (Point, Point) DetectBothEyes(Mat face, CascadeClassifier eyeCascade, ref Rect? searchedLeftEye, ref Rect? searchedRightEye)
        {
            const float EYE_SX = 0.16f;
            const float EYE_SY = 0.26f;
            const float EYE_SW = 0.30f;
            const float EYE_SH = 0.28f;

            int leftX = (int)Math.Round(face.Cols * EYE_SX);
            int topY = (int)Math.Round(face.Rows * EYE_SY);
            int widthX = (int)Math.Round(face.Cols * EYE_SW);
            int heightY = (int)Math.Round(face.Rows * EYE_SH);
            int rightX = (int)Math.Round(face.Cols * (1.0 - EYE_SX - EYE_SW));  // Start of right-eye corner

            Mat topLeftOfFace = new Mat(face,new Rect(leftX, topY, widthX, heightY));
            Mat topRightOfFace = new Mat(face, new Rect(rightX, topY, widthX, heightY));
            Rect leftEyeRect, rightEyeRect = new Rect();

            if(searchedLeftEye is not null)
                searchedLeftEye = new Rect(leftX, topY, widthX, heightY);
            if (searchedRightEye is not null)
                rightEyeRect = new Rect(rightX, topY, widthX, heightY);

            leftEyeRect = ObjectDetection.DetectLargestObject(topLeftOfFace, eyeCascade, topLeftOfFace.Cols);
            rightEyeRect = ObjectDetection.DetectLargestObject(topRightOfFace, eyeCascade, topRightOfFace.Cols);


            Point leftEye = new Point();
            Point rightEye = new Point();
            if (leftEyeRect.Width > 0)
            {   // Check if the eye was detected.
                leftEyeRect.X += leftX;    // Adjust the left-eye rectangle because the face border was removed.
                leftEyeRect.Y += topY;
                leftEye = new Point(leftEyeRect.X + leftEyeRect.Width / 2, leftEyeRect.Y + leftEyeRect.Height / 2);
            }
            else
            {
                leftEye = new Point(-1, -1);    // Return an invalid point
            }

            if (rightEyeRect.Width > 0)
            { // Check if the eye was detected.
                rightEyeRect.X += rightX; // Adjust the right-eye rectangle, since it starts on the right side of the image.
                rightEyeRect.Y += topY;  // Adjust the right-eye rectangle because the face border was removed.
                rightEye = new Point(rightEyeRect.X + rightEyeRect.Width / 2, rightEyeRect.Y + rightEyeRect.Height / 2);
            }
            else
            {
                rightEye = new Point(-1, -1);    // Return an invalid point
            }

            return (leftEye, rightEye);
        }


        public static Mat GetPreProcessedFace(Mat srcImg, int desiredFaceWidth, CascadeClassifier faceCascade, CascadeClassifier eyeCascade, bool doLeftAndRightSeparately,
            Rect storeFaceRect, Point storeLeftEye, Point storeRightEye,ref Rect? searchedLeftEye,ref Rect? searchedRightEye)
        {
            // Use square faces.
            int desiredFaceHeight = desiredFaceWidth;

            // Mark the detected face region and eye search regions as invalid, in case they aren't detected.

            //if (storeFaceRect is not null)
            //    storeFaceRect.Width = -1;
            //if (storeLeftEye is not null)
            //    storeLeftEye.X = -1;
            //if (storeRightEye is not null)
            //    storeRightEye.X = -1;
            //if (searchedLeftEye is not null)
            //    searchedLeftEye.Width = -1;
            //if (searchedRightEye is not null)
            //    searchedRightEye.Width = -1;

            Rect faceRect = new Rect();
            faceRect = ObjectDetection.DetectLargestObject(srcImg, faceCascade, srcImg.Cols);
            // Check if a face was detected.
            if (faceRect.Width > 0)
            {

                // Give the face rect to the caller if desired.
                //if (storeFaceRect)
                //    *storeFaceRect = faceRect;

                var faceImage = srcImg.ToImage<Gray, byte>();
                faceImage.ROI = faceRect;
                Mat faceImg = faceImage.Mat;
                return faceImg;
                // If the input image is not grayscale, then convert the BGR or BGRA color image to grayscale.
                Mat gray = new Mat();
                if (faceImg.NumberOfChannels == 3)
                {
                    CvInvoke.CvtColor(faceImg, gray, ColorConversion.Bgr2Gray);
                }
                else if (faceImg.NumberOfChannels == 4)
                {
                    CvInvoke.CvtColor(faceImg, gray, ColorConversion.Bgra2Gray);
                }
                else
                {
                    // Access the input image directly, since it is already grayscale.
                    gray = faceImg;
                }

                (Point, Point) eyes;
                eyes = DetectBothEyes(gray, eyeCascade, ref searchedLeftEye, ref searchedRightEye);


                var leftEye = eyes.Item1;
                var rightEye = eyes.Item2;
                // Check if both eyes were detected.
                if(leftEye.X >= 0 && rightEye.X >= 0)
                {

                    // Make the face image the same size as the training images.

                    // Since we found both eyes, lets rotate & scale & translate the face so that the 2 eyes
                    // line up perfectly with ideal eye positions. This makes sure that eyes will be horizontal,
                    // and not too far left or right of the face, etc.

                    // Get the center between the 2 eyes.
                    PointF eyesCenter = new PointF((leftEye.X + rightEye.X) * 0.5f, (leftEye.Y + rightEye.Y) * 0.5f);
                    // Get the angle between the 2 eyes.
                    double dy = (rightEye.Y - leftEye.Y);
                    double dx = (rightEye.X - leftEye.X);
                    double len = Math.Sqrt(dx * dx + dy * dy);
                    double angle = Math.Atan2(dy, dx) * 180.0 / Math.PI; // Convert from radians to degrees.

                    // Hand measurements shown that the left eye center should ideally be at roughly (0.19, 0.14) of a scaled face image.
                    const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);
                    // Get the amount we need to scale the image to be the desired fixed size we want.
                    double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * desiredFaceWidth;
                    double scale = desiredLen / len;
                    // Get the transformation matrix for rotating and scaling the face to the desired angle & size.
                    Mat rot_mat = new Mat();
                    CvInvoke.GetRotationMatrix2D(eyesCenter, angle, scale,rot_mat);
                    // Shift the center of the eyes to be the desired center between the eyes.
                    var centerX = rot_mat.GetDoubleValue(0, 2);
                    centerX += desiredFaceWidth * 0.5f - eyesCenter.X;
                    rot_mat.SetDoubleValue(0,2,centerX);
                    var centerY = rot_mat.GetDoubleValue(1, 2);
                    centerY += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.Y;
                    rot_mat.SetDoubleValue(1, 2, centerY);

                    // Rotate and scale and translate the image to the desired angle & size & position!
                    // Note that we use 'w' for the height instead of 'h', because the input face has 1:1 aspect ratio.
                    Mat warped = new Mat(desiredFaceHeight, desiredFaceWidth, DepthType.Cv8U,1); // Clear the output image to a default grey.
                    CvInvoke.WarpAffine(gray, warped, rot_mat, warped.Size);
                    //CvInvoke.Imshow("warped", warped);
                    //return warped;
                    // Give the image a standard brightness and contrast, in case it was too dark or had low contrast.
                    if (!doLeftAndRightSeparately)
                    {
                        // Do it on the whole face.
                       CvInvoke.EqualizeHist(warped, warped);
                    }
                    else
                    {
                        // Do it seperately for the left and right sides of the face.
                        warped = EqualizeLeftAndRightHalves(warped);
                    }
                    //CvInvoke.Imshow("equalized", warped);
                    return warped;
                    // Use the "Bilateral Filter" to reduce pixel noise by smoothing the image, but keeping the sharp edges in the face.
                    Mat filtered = new Mat(warped.Size, DepthType.Cv8U,1,warped.DataPointer,warped.Step);
                        //CvInvoke.BilateralFilter(warped, filtered, 0, 10.0, 2.0);
                        //CvInvoke.Imshow("filtered", filtered);

                        // Filter out the corners of the face, since we mainly just care about the middle parts.
                        // Draw a filled ellipse in the middle of the face-sized image.
                        Mat mask = new Mat(warped.Size, DepthType.Cv8U, 1, warped.DataPointer, warped.Step); // Start with an empty mask.
                        Point faceCenter = new Point(desiredFaceWidth / 2, (int)Math.Round(desiredFaceHeight * FACE_ELLIPSE_CY));
                        Size size = new Size((int)Math.Round(desiredFaceWidth * FACE_ELLIPSE_W), (int)Math.Round(desiredFaceHeight * FACE_ELLIPSE_H));

                        //CvInvoke.Ellipse(mask, faceCenter, size, 0, 0, 360, new Bgr(System.Drawing.Color.Aqua).MCvScalar, 2);
                        //CvInvoke.Imshow("mask", mask);

                        // Use the mask, to remove outside pixels.
                        Mat dstImg = new Mat(warped.Size, DepthType.Cv8U, 1, warped.DataPointer, warped.Step); // Clear the output image to a default gray.
                        /*
                        namedWindow("filtered");
                        imshow("filtered", filtered);
                        namedWindow("dstImg");
                        imshow("dstImg", dstImg);
                        namedWindow("mask");
                        imshow("mask", mask);
                        */
                        // Apply the elliptical mask on the face.
                        filtered.CopyTo(dstImg, mask);  // Copies non-masked pixels from filtered to dstImg.
                                                        //imshow("dstImg", dstImg);

                    //CvInvoke.Imshow("dstImg",dstImg);
                        return dstImg;

                }



            }
                return new Mat();
        }


        public static Mat EqualizeLeftAndRightHalves(Mat faceImg)
        {
            // It is common that there is stronger light from one half of the face than the other. In that case,
            // if you simply did histogram equalization on the whole face then it would make one half dark and
            // one half bright. So we will do histogram equalization separately on each face half, so they will
            // both look similar on average. But this would cause a sharp edge in the middle of the face, because
            // the left half and right half would be suddenly different. So we also histogram equalize the whole
            // image, and in the middle part we blend the 3 images together for a smooth brightness transition.

            int w = faceImg.Cols;
            int h = faceImg.Rows;

            // 1) First, equalize the whole face.
            Mat wholeFace = new Mat();
            CvInvoke.EqualizeHist(faceImg, wholeFace);

            // 2) Equalize the left half and the right half of the face separately.
            int midX = w / 2;
            Image<Gray, byte> _faceImage = faceImg.ToImage<Gray, byte>();
            _faceImage.ROI = new Rect(0, 0, midX, h);
            Mat leftSide = _faceImage.Mat;
            _faceImage = faceImg.ToImage<Gray, byte>();
            _faceImage.ROI = new Rect(0, 0, midX, h);
            Mat rightSide = _faceImage.Mat;
            CvInvoke.EqualizeHist(leftSide, leftSide);
            CvInvoke.EqualizeHist(rightSide, rightSide);

            // 3) Combine the left half and right half and whole face together, so that it has a smooth transition.
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int v;
                    if (x < w / 4)
                    {          // Left 25%: just use the left face.
                        //v = leftSide.at<uchar>(y, x);
                        v = (int)leftSide.GetDoubleValue(y, x);
                    }
                    else if (x < w * 2 / 4)
                    {   // Mid-left 25%: blend the left face & whole face.
                        int lv = (int)leftSide.GetDoubleValue(y, x);
                        int wv = (int)wholeFace.GetDoubleValue(y, x); ;
                        // Blend more of the whole face as it moves further right along the face.
                        float f = (x - w * 1 / 4) / (float)(w * 0.25f);
                        v = (int)Math.Round((1.0f - f) * lv + (f) * wv);
                    }
                    else if (x < w * 3 / 4)
                    {   // Mid-right 25%: blend the right face & whole face.
                        int rv = (int)rightSide.GetDoubleValue(y, x - midX);
                        int wv = (int)wholeFace.GetDoubleValue(y, x);
                        // Blend more of the right-side face as it moves further right along the face.
                        float f = (x - w * 2 / 4) / (float)(w * 0.25f);
                        v = (int)Math.Round((1.0f - f) * wv + (f) * rv);
                    }
                    else
                    {                  // Right 25%: just use the right face.
                        v = (int)rightSide.GetDoubleValue(y, x - midX);
                    }
                    faceImg.SetDoubleValue(y, x, v);
                }// end x loop
            }//end y loop
            return faceImg;
        }

    }
}
