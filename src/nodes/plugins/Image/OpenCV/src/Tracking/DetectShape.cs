using System.ComponentModel.Composition;
using System.Drawing;
using Emgu.CV.GPU;
using VVVV.PluginInterfaces.V2;
using VVVV.Utils.VMath;
using VVVV.Core.Logging;
using Emgu.CV;
using Emgu.CV.Structure;
using System.Collections.Generic;
using System;
using VVVV.CV.Core;

namespace VVVV.CV.Nodes.Tracking
{
    public enum ShapeType { None, Circle, Triangle, Rectangle };

    public class TrackingShape
	{
        public ShapeType Type = ShapeType.None;
		public Vector2D Position;
		public Vector2D Scale;
        public double Angle = 0;
	}
    
	public class TrackingShapeInstance : IDestinationInstance
	{
        public bool Enabled = true;

		//private readonly Vector2D FMinimumSourceXY = new Vector2D(0, 0);
		//private readonly Vector2D FMinimumDestXY = new Vector2D(-0.5, 0.5);
		//private readonly Vector2D FMaximumDestXY = new Vector2D(0.5, -0.5);

		//private CascadeClassifier FCascadeClassifier;
		//private GpuCascadeClassifier FGpuCascadeClassifier;

		private readonly CVImage FGrayScale = new CVImage();
		private readonly List<TrackingShape> FTrackingShapes = new List<TrackingShape>();

        #region tracking params
        public double CannyThreshold { get; set; }
        public double CannyThresholdLinking { get; set; }
        public double CircleAccumulatorThreshold { get; set; }
        public double CircleAccumulatorResolution { get; set; }
        public double CircleMinDistance { get; set; }
        public int CircleMinRadius { get; set; }
        public int CircleMaxRadius { get; set; }
        public double EdgeDistance { get; set; }
        public double EdgeAngle { get; set; }
        public int EdgeThreshold { get; set; }
        public double EdgeMinWidth { get; set; }
        public double EdgeGap { get; set; }
        #endregion

        public TrackingShapeInstance()
		{
            CannyThreshold = 180.0;
            CannyThresholdLinking = 120.0;
            CircleAccumulatorThreshold = 120.0;
            CircleAccumulatorResolution = 2.0;
            CircleMinDistance = 20.0;
            CircleMinRadius = 5;
            CircleMaxRadius = 0;
            EdgeDistance = 1.0;
            EdgeAngle = Math.PI/45.0;
            EdgeThreshold = 20;
            EdgeMinWidth = 30.0;
            EdgeGap = 10.0;
        }

		public List<TrackingShape> TrackingShapes
		{
			get { return FTrackingShapes; }
		}

		//public void LoadHaarCascade(string path)
		//{
		//	FCascadeClassifier = new CascadeClassifier(path);
		//	FGpuCascadeClassifier = new GpuCascadeClassifier(path);
		//}

		public override void Allocate()
		{
			FGrayScale.Initialise(FInput.Image.ImageAttributes.Size, TColorFormat.L8);
		}

		public override void Process()
		{
            if (!Enabled)
                return;
            
            FInput.Image.GetImage(TColorFormat.L8, FGrayScale);
			var grayImage = FGrayScale.GetImage() as Image<Gray, byte>;
            if(grayImage != null)
            {
                // do this by erode & dilate node
                //grayImage = grayImage.PyrDown().PyrUp();

                // circle detection
                CircleF[] circles = grayImage.HoughCircles(
                    new Gray(CannyThreshold),
                    new Gray(CircleAccumulatorThreshold),
                    CircleAccumulatorResolution,    //  resolution of the accumulator used to detect centers of the circles
                    CircleMinDistance,              //  min distance
                    CircleMinRadius,                //  min radius
                    CircleMaxRadius                 //  max radius
                    )[0];                           //  get the circles from the first channel

                // canny and edge detection
                Image<Gray, byte> cannyEdges = grayImage.Canny(CannyThreshold, CannyThresholdLinking);
                LineSegment2D[] lines = cannyEdges.HoughLinesBinary(
                    EdgeDistance,       //  distance resolution in pixel-related units
                    EdgeAngle,          //  angle resolution measured in radians
                    EdgeThreshold,      //  threshold
                    EdgeMinWidth,       //  min line width
                    EdgeGap             //  gap between lines
                    )[0];               //  get the lines from the first channel

                // find triangles and rectangles
                List<Triangle2DF> triangleList = new List<Triangle2DF>();
                List<MCvBox2D> boxList = new List<MCvBox2D>(); //a box is a rotated rectangle
                using (MemStorage storage = new MemStorage()) //allocate storage for contour approximation
                {
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
                }



                // clear all
                FTrackingShapes.Clear();



                // circle
                foreach (CircleF circle in circles)
                {
                    TrackingShape trackingShape = new TrackingShape();
                    trackingShape.Type = ShapeType.Circle;
                    trackingShape.Position = new Vector2D(circle.Center.X, circle.Center.Y);
                    trackingShape.Scale = new Vector2D(circle.Radius * 2);

                    TrackingShapes.Add(trackingShape);
                }

                // triangle
                foreach (Triangle2DF triangle in triangleList)
                {
                    TrackingShape trackingShape = new TrackingShape();
                    trackingShape.Type = ShapeType.Triangle;
                    trackingShape.Position = new Vector2D(triangle.Centeroid.X, triangle.Centeroid.Y);

                    float xmin = Math.Min(triangle.V0.X, Math.Min(triangle.V1.X, triangle.V2.X));
                    float xmax = Math.Max(triangle.V0.X, Math.Max(triangle.V1.X, triangle.V2.X));
                    float ymin = Math.Min(triangle.V0.Y, Math.Min(triangle.V1.Y, triangle.V2.Y));
                    float ymax = Math.Max(triangle.V0.Y, Math.Min(triangle.V1.Y, triangle.V2.Y));
                    
                    trackingShape.Scale = new Vector2D(xmax - xmin, ymax - ymin);
                    
                    // calcuate angle?

                    TrackingShapes.Add(trackingShape);
                }

                // rectangle
                foreach (MCvBox2D box in boxList)
                {
                    TrackingShape trackingShape = new TrackingShape();
                    trackingShape.Type = ShapeType.Rectangle;
                    trackingShape.Position = new Vector2D(box.center.X, box.center.Y);
                    trackingShape.Scale = new Vector2D(box.size.Width, box.size.Height);
                    trackingShape.Angle = box.angle;

                    TrackingShapes.Add(trackingShape);
                }
            }

			//if (GpuInvoke.HasCuda && AllowGpu)
			//{
			//	rectangles = ProcessOnGpu(grayImage);
			//}
			//else
			//{
			//	rectangles = ProcessOnCpu(grayImage);
			//}

			//FTrackingObjects.Clear();
			//foreach (var rectangle in rectangles)
			//{
			//	var trackingObject = new TrackingObject();

			//	var center = new Vector2D(rectangle.X + rectangle.Width / 2, rectangle.Y + rectangle.Height / 2);
			//	var maximumSourceXY = new Vector2D(FGrayScale.Width, FGrayScale.Height);

			//	trackingObject.Position = VMath.Map(center, FMinimumSourceXY, maximumSourceXY, FMinimumDestXY,
			//										FMaximumDestXY, TMapMode.Float);
			//	trackingObject.Scale = VMath.Map(new Vector2D(rectangle.Width, rectangle.Height), FMinimumSourceXY.x, maximumSourceXY.x, 0,
			//									 1, TMapMode.Float);

			//	FTrackingObjects.Add(trackingObject);
			//}
		}

        /*
		private Rectangle[] ProcessOnGpu(Image<Gray, byte> grayImage)
		{
			if (FGpuCascadeClassifier == null)
			{
				Status = "Can't load Haar file";
				return new Rectangle[0];
			}

			using (var gpuImage = new GpuImage<Gray, byte>(grayImage))
			{
				return FGpuCascadeClassifier.DetectMultiScale(gpuImage, ScaleFactor, MinNeighbors, MinSize);
			}
		}

		private Rectangle[] ProcessOnCpu(Image<Gray, byte> grayImage)
		{
			if (FCascadeClassifier == null)
			{
				Status = "Can't load Haar file";
				return new Rectangle[0];
			}

			

			if (grayImage == null)
			{
				Status = "Can't get image or convert it to grayscale";
				return new Rectangle[0];
			}

			grayImage._EqualizeHist();

			return FCascadeClassifier.DetectMultiScale(grayImage, ScaleFactor, MinNeighbors, MinSize, MaxSize);

			
		}
        */
	}
    

    [PluginInfo(Name = "DetectShape", Category = "CV.Image", Help = "Tracks simple shapes", Author = "mino", Credits = "elliotwoods", Tags = "tracking, shape")]
	public class ShapeTrackingNode : IDestinationNode<TrackingShapeInstance>
	{
        #region fields & pins
        [Input("Canny Threshold", DefaultValue = 180.0)]
        private IDiffSpread<double> FCannyThresholdIn;

        [Input("Canny Threshold Linking", DefaultValue = 120.0)]
        private IDiffSpread<double> FCannyThresholdLinkingIn;

        [Input("Circle Accumulator Threshold", DefaultValue = 120.0)]
        private IDiffSpread<double> FCircleAccumulatorThresholdIn;

        [Input("Circle Accumulator Resolution", DefaultValue = 2.0)]
        private IDiffSpread<double> FCircleAccumulatorResolutionIn;

        [Input("Circle Min Distance", DefaultValue = 20.0)]
        private IDiffSpread<double> FCircleMinDistanceIn;

        [Input("Circle Min Radius", DefaultValue = 5)]
        private IDiffSpread<int> FCircleMinRadiusIn;

        [Input("Circle Max Radius", DefaultValue = 0)]
        private IDiffSpread<int> FCircleMaxRadiusIn;

        [Input("Edge Distance (px)", DefaultValue = 1.0)]
        private IDiffSpread<double> FEdgeDistanceIn;

        [Input("Edge Angle (rad)", DefaultValue = Math.PI/45.0)]
        private IDiffSpread<double> FEdgeAngleIn;

        [Input("Edge Threshold", DefaultValue = 20)]
        private IDiffSpread<int> FEdgeThresholdIn;

        [Input("Edge Min Width", DefaultValue = 30.0)]
        private IDiffSpread<double> FEdgeMinWidthIn;

        [Input("Edge Gap", DefaultValue = 10.0)]
        private IDiffSpread<double> FEdgeGapIn;

        [Input("Enabled", DefaultValue = 1)]
        private IDiffSpread<bool> FPinInEnabled;

        [Output("Type")]
        private ISpread<ISpread<string>> FTypeOut;

        [Output("Position")] 
		private ISpread<ISpread<Vector2D>> FPositionOut;

		[Output("Scale")] 
		private ISpread<ISpread<Vector2D>> FScaleOut;

        [Output("Angle")]
        private ISpread<ISpread<double>> FAngleOut;

        [Output("Status")] 
		private ISpread<string> FStatusOut;

        [Import] 
		private ILogger FLogger;
		#endregion fields & pins

		protected override void Update(int instanceCount, bool spreadChanged)
		{
			FStatusOut.SliceCount = instanceCount;
			CheckParams(instanceCount);
			Output(instanceCount);
		}

		private void CheckParams(int instanceCount)
		{
            int i;

            if (FCannyThresholdIn.IsChanged)
                for (i = 0; i < instanceCount; i++)
                    FProcessor[i].CannyThreshold = FCannyThresholdIn[i];

            if (FCannyThresholdLinkingIn.IsChanged)
                for (i = 0; i<instanceCount; i++)
                    FProcessor[i].CannyThresholdLinking = FCannyThresholdLinkingIn[i];
        
            if (FCircleAccumulatorThresholdIn.IsChanged)
                for (i = 0; i<instanceCount; i++)
                    FProcessor[i].CircleAccumulatorThreshold = FCircleAccumulatorThresholdIn[i];
        
            if (FCircleAccumulatorResolutionIn.IsChanged)
                for (i = 0; i<instanceCount; i++)
                    FProcessor[i].CircleAccumulatorResolution = FCircleAccumulatorResolutionIn[i];
        
            if (FCircleMinDistanceIn.IsChanged)
                for (i = 0; i<instanceCount; i++)
                    FProcessor[i].CircleMinDistance = FCircleMinDistanceIn[i];
        
            if (FCircleMinRadiusIn.IsChanged)
                for (i = 0; i<instanceCount; i++)
                    FProcessor[i].CircleMinRadius = FCircleMinRadiusIn[i];
        
            if (FCircleMaxRadiusIn.IsChanged)
                for (i = 0; i<instanceCount; i++)
                    FProcessor[i].CircleMaxRadius = FCircleMaxRadiusIn[i];
        
            if (FEdgeDistanceIn.IsChanged)
                for (i = 0; i<instanceCount; i++)
                    FProcessor[i].EdgeDistance = FEdgeDistanceIn[i];
        
            if (FEdgeAngleIn.IsChanged)
                for (i = 0; i<instanceCount; i++)
                    FProcessor[i].EdgeAngle = FEdgeAngleIn[i];
        
            if (FEdgeThresholdIn.IsChanged)
                for (i = 0; i<instanceCount; i++)
                    FProcessor[i].EdgeThreshold = FEdgeThresholdIn[i];
        
            if (FEdgeMinWidthIn.IsChanged)
                for (i = 0; i<instanceCount; i++)
                    FProcessor[i].EdgeMinWidth = FEdgeMinWidthIn[i];
        
            if (FEdgeGapIn.IsChanged)
                for (i = 0; i<instanceCount; i++)
                    FProcessor[i].EdgeGap = FEdgeGapIn[i];

            if (FPinInEnabled.IsChanged)
                for (i = 0; i < instanceCount; i++)
                    FProcessor[i].Enabled = FPinInEnabled[i];
        }

		private void Output(int instanceCount)
		{
            FTypeOut.SliceCount = instanceCount;
			FPositionOut.SliceCount = instanceCount;
			FScaleOut.SliceCount = instanceCount;
            FAngleOut.SliceCount = instanceCount;

            for (int i = 0; i < instanceCount; i++)
            {
                var count = FProcessor[i].TrackingShapes.Count;
                FTypeOut[i].SliceCount = count;
                FPositionOut[i].SliceCount = count;
                FScaleOut[i].SliceCount = count;
                FAngleOut[i].SliceCount = count;

                for (int j = 0; j < count; j++)
                {
                    try
                    {
                        //FTypeOut[i][j] = FProcessor[i].TrackingShapes[j].Type;
                        if (FProcessor[i].TrackingShapes[j].Type == ShapeType.Circle)
                            FTypeOut[i][j] = "Circle";
                        else if (FProcessor[i].TrackingShapes[j].Type == ShapeType.Triangle)
                            FTypeOut[i][j] = "Triangle";
                        else if (FProcessor[i].TrackingShapes[j].Type == ShapeType.Rectangle)
                            FTypeOut[i][j] = "Rectangle";
                        else
                            FTypeOut[i][j] = "";
                        FPositionOut[i][j] = FProcessor[i].TrackingShapes[j].Position;
                        FScaleOut[i][j] = FProcessor[i].TrackingShapes[j].Scale;
                        FAngleOut[i][j] = FProcessor[i].TrackingShapes[j].Angle;
                    }
                    catch
                    {
                        FLogger.Log(LogType.Error, "Desync in threads");
                    }
                }
            }
        }
	}
}