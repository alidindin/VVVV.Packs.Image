#region using
using System.Collections.Generic;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using VVVV.PluginInterfaces.V2;
using VVVV.Utils.VMath;
using System;
using VVVV.Utils.VColor;
using VVVV.CV.Core;

#endregion

namespace VVVV.CV.Nodes
{
    [FilterInstance("DrawShapeDetection", Help = "")]
	public class DrawShapeDetectionInstance : IFilterInstance
	{
        //Gray cannyThreshold = new Gray(180);
        //Gray cannyThresholdLinking = new Gray(120);
        //Gray circleAccumulatorThreshold = new Gray(120);
        //CVImage FGrayscale = new CVImage();

        //	//if changing these properties means we need to change the output image
        //	//size or colour type, then we need to call
        //	//Allocate();
        //}

        public override void Allocate()
		{
            //This function gets called whenever the output image needs to be initialised
            //Initialising = setting the attributes (i.e. setting the image header and allocating the memory)

            //FGrayscale.Initialise(FInput.ImageAttributes.Size, TColorFormat.L8);
            FOutput.Image.Initialise(400, 400, TColorFormat.RGB8);
		}

		public override void Process()
		{
            //If we want to pull out an image in a specific format
            //then we must have a local instance of a CVImage initialised to that format
            //and use
            //FInput.Image.GetImage(TColorFormat.L8, FInputL8);
            //in that example, we expect to have a FInputL8 locally which has been intialised
            //with the correct size and colour format


            //Whenever you access the pixels directly of FInput
            //e.g. when using the .CvMat accessor
            //you must lock it for reading using 
            //if (!FInput.LockForReading()) //this
            //	return;

            Image<Rgb, byte> img = FInput.Image.GetImage() as Image<Rgb, byte>;
            img = img.Resize(400, 400, INTER.CV_INTER_LINEAR, true);

            //FInput.Image.GetImage(TColorFormat.L8, FGrayscale);

            //FInput.ReleaseForReading(); //and  this after you've finished with FImage

            // copy sample of emgucv
            //Image<Gray, byte> gray = FGrayscale.GetImage() as Image<Gray, byte>;
            Image<Gray, byte> gray = img.Convert<Gray, byte>();
            gray = gray.PyrDown().PyrUp();

            // circle detection
            double cannyThreshold = 180.0;
            double circleAccumulatorThreshold = 120;
            CircleF[] circles = gray.HoughCircles(
                new Gray(cannyThreshold),
                new Gray(circleAccumulatorThreshold),
                2.0,
                20.0,
                5,
                0
                )[0];

            // canny and edge detection
            double cannyThresholdLinking = 120.0;
            Image<Gray, byte> cannyEdges = gray.Canny(cannyThreshold, cannyThresholdLinking);
            LineSegment2D[] lines = cannyEdges.HoughLinesBinary(
                1,
                Math.PI / 45.0,
                20,
                30,
                10
                )[0];

            // find triangles and rectangles
            List<Triangle2DF> triangleList = new List<Triangle2DF>();
            List<MCvBox2D> boxList = new List<MCvBox2D>(); //a box is a rotated rectangle
            using (MemStorage storage = new MemStorage()) //allocate storage for contour approximation
                for (
                   Contour<Point> contours = cannyEdges.FindContours(
                      Emgu.CV.CvEnum.CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE,
                      Emgu.CV.CvEnum.RETR_TYPE.CV_RETR_LIST,
                      storage);
                   contours != null;
                   contours = contours.HNext)
                {
                    Contour<Point> currentContour = contours.ApproxPoly(contours.Perimeter * 0.05, storage);

                    if (currentContour.Area > 250) //only consider contours with area greater than 250
                    {
                        if (currentContour.Total == 3) //The contour has 3 vertices, it is a triangle
                        {
                            Point[] pts = currentContour.ToArray();
                            triangleList.Add(new Triangle2DF(
                               pts[0],
                               pts[1],
                               pts[2]
                               ));
                        }
                        else if (currentContour.Total == 4) //The contour has 4 vertices.
                        {
                            #region determine if all the angles in the contour are within [80, 100] degree
                            bool isRectangle = true;
                            Point[] pts = currentContour.ToArray();
                            LineSegment2D[] edges = PointCollection.PolyLine(pts, true);

                            for (int i = 0; i < edges.Length; i++)
                            {
                                double angle = Math.Abs(
                                   edges[(i + 1) % edges.Length].GetExteriorAngleDegree(edges[i]));
                                if (angle < 80 || angle > 100)
                                {
                                    isRectangle = false;
                                    break;
                                }
                            }
                            #endregion

                            if (isRectangle) boxList.Add(currentContour.GetMinAreaRect());
                        }
                    }
                }

            // input as background
            //FOutput.Image.SetImage(FInput.Image);

            //FInput.Image.GetImage(TColorFormat.RGB8, inputImage1);

            #region draw triangles and rectangles
            //Image<Bgr, Byte> triangleRectangleImage = img.CopyBlank();
            foreach (Triangle2DF triangle in triangleList)
                img.Draw(triangle, new Rgb(0, 0, 255), 2);
            foreach (MCvBox2D box in boxList)
                img.Draw(box, new Rgb(0, 255, 0), 2);
            #endregion

            // draw circle
            //Image<Bgr, Byte> circleImage = inputImage.CopyBlank();
            foreach (CircleF circle in circles)
                img.Draw(circle, new Rgb(0, 0, 255), 2);
            //FOutput.Image.SetImage(circleImage);

            #region draw lines
            // Image<Bgr, Byte> lineImage = img.CopyBlank();
            foreach (LineSegment2D line in lines)
                img.Draw(line, new Rgb(255, 0, 0), 2);
            //lineImageBox.Image = lineImage;
            #endregion

            FOutput.Image.SetImage(img);

			FOutput.Send();
		}
	}
}
