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
    [FilterInstance("DrawContour", Help = "")]
	public class DrawContourInstance : IFilterInstance
	{
        // test
        [Input("Index")]
        public int Index = 0;
        private int counter = 0;

        [Input("Max Level")]
        public int MaxLevel = 0;

        private int FThickness = 3;
        [Input("Thickness", MinValue = -1, DefaultValue = 1)]
        public int Thickness
        {
            set
            {
                if (value < -1)
                    value = -1;

                //if (value > 7)
                //    value = 7;

                FThickness = value;
            }
        }

        CVImage FGrayscale = new CVImage();

        //	//if changing these properties means we need to change the output image
        //	//size or colour type, then we need to call
        //	//Allocate();
        //}

        public override void Allocate()
		{
            //This function gets called whenever the output image needs to be initialised
            //Initialising = setting the attributes (i.e. setting the image header and allocating the memory)
            
            FGrayscale.Initialise(FInput.ImageAttributes.Size, TColorFormat.L8);
            FOutput.Image.Initialise(FInput.ImageAttributes);
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
			if (!FInput.LockForReading()) //this
				return;

            FInput.Image.GetImage(TColorFormat.L8, FGrayscale);

            FInput.ReleaseForReading(); //and  this after you've finished with FImage

            // 1) input as background
            //FOutput.Image.SetImage(FInput.Image);

            // 2) black as background
            Image<Gray, byte> blackImage = new Image<Gray, byte>(FInput.ImageAttributes.Width, FInput.ImageAttributes.Height, new Gray());
            FOutput.Image.SetImage(blackImage);

            Image<Gray, byte> img = FGrayscale.GetImage() as Image<Gray, byte>;
            if (img != null)
            {
                using (MemStorage storage = new MemStorage())
                {
                    counter = 0;

                    for (Contour<Point> contours = img.FindContours(CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_NONE, RETR_TYPE.CV_RETR_TREE, storage); contours != null; contours = contours.HNext)
                    {
                        if (Index == -1 || Index == counter)
                        {
                            CvInvoke.cvDrawContours(FOutput.CvMat, contours, new MCvScalar(255), new MCvScalar(255), MaxLevel, FThickness, LINE_TYPE.CV_AA, new Point(0, 0));
                        }
                        counter++;
                    }
                }
            }

            //FInput.ReleaseForReading(); //and  this after you've finished with FImage
			
			FOutput.Send();
		}
	}
}
