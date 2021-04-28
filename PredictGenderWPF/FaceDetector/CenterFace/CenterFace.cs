using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NcnnDotNet;
using System.Drawing;

namespace FaceDetector.CenterFace
{
    /// <summary>
    /// Provides the method to find face methods. This class cannot be inherited.
    /// </summary>
    public sealed class CenterFace : DisposableObject
    {
        #region Fields
        private readonly CenterFaceParameter _centerFaceParameter;
        private readonly CenterFace _ultraFaceDetector;
        private readonly Net _Net;

        private int _DH;

        private int _DW;

        private float _DScaleH;

        private float _DScaleW;

        private float _ScaleH;

        private float _ScaleW;

        private int _ImageH;

        private int _ImageW;

        public string DetectorName => "CenterFace";

        public string BinPath => @"C:\Users\fzhil\source\repos\CropFacesWithCenterFace\model\centerface.bin";
        public string ParamPath => @"C:\Users\fzhil\source\repos\CropFacesWithCenterFace\model\centerface.param";

        public Pen DrawPen => new Pen(Color.FromArgb(255, 20, 255), 3);

        #endregion

        #region Constructors
        public CenterFace()
        {
            _centerFaceParameter = new CenterFaceParameter
            {
                BinFilePath = BinPath,
                ParamFilePath = ParamPath,
            };

            _ultraFaceDetector = Create(_centerFaceParameter);
        }

        private CenterFace(CenterFaceParameter parameter)
        {
            this._Net = new Net();
            this._Net.LoadParam(parameter.ParamFilePath);
            this._Net.LoadModel(parameter.BinFilePath);
        }

        #endregion

        #region Methods

        /// <summary>
        /// Create a new instance of the <see cref="CenterFace"/> class with the specified parameter.
        /// </summary>
        /// <param name="parameter">The parameter.</param>
        /// <returns>The <see cref="CenterFace"/> this method creates.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="parameter"/> is null.</exception>
        /// <exception cref="ArgumentException">The model binary file is null or whitespace. Or the param file is null or whitespace.</exception>
        /// <exception cref="FileNotFoundException">The model binary file is not found. Or the param file is not found.</exception>
        public static CenterFace Create(CenterFaceParameter parameter)
        {
            if (parameter == null)
            {
                throw new ArgumentNullException(nameof(parameter));
            }
            if (string.IsNullOrWhiteSpace(parameter.BinFilePath))
            {
                throw new ArgumentException("The model binary file is null or whitespace", nameof(parameter.BinFilePath));
            }
            if (string.IsNullOrWhiteSpace(parameter.ParamFilePath))
            {
                throw new ArgumentException("The param file is null or whitespace", nameof(parameter.ParamFilePath));
            }
            if (!File.Exists(parameter.BinFilePath))
            {
                throw new FileNotFoundException("The model binary file is not found.");
            }
            if (!File.Exists(parameter.ParamFilePath))
            {
                throw new FileNotFoundException("The param file is not found.");
            }

            return new CenterFace(parameter);
        }

        /// <summary>
        /// Returns an enumerable collection of face location correspond to all faces in specified image.
        /// </summary>
        /// <param name="image">The image contains faces. The image can contain multiple faces.</param>
        /// <param name="resizedWidth">The pixel width after resized input image.</param>
        /// <param name="resizedHeight">The pixel height after resized input image.</param>
        /// <param name="scoreThreshold">The score threshold for detecting face. The default is 0.5f.</param>
        /// <param name="nmsThreshold">The non maximum suppression threshold for detecting face. The default is 0.3f.</param>
        /// <returns>An enumerable collection of face location correspond to all faces in specified image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="image"/> is empty.</exception>
        /// <exception cref="ObjectDisposedException"><paramref name="image"/> or this object is disposed.</exception>
        public IEnumerable<FaceInfo> Detect(Mat image,
                                            int resizedWidth,
                                            int resizedHeight,
                                            float scoreThreshold = 0.5f,
                                            float nmsThreshold = 0.3f)
        {
            if (image == null)
            {
                throw new ArgumentNullException(nameof(image));
            }

            image.ThrowIfDisposed();
            this.ThrowIfDisposed();

            List<FaceInfo> faceList = new List<FaceInfo>();
            if (this.Detect(image,
                            resizedWidth,
                            resizedHeight,
                            scoreThreshold,
                            nmsThreshold,
                            faceList) != 0)
            {
                throw new ArgumentException("Image is empty.", nameof(image));
            }

            return faceList.ToArray();
        }

        #region Overrides 

        /// <summary>
        /// Releases all unmanaged resources.
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();

            this._Net?.Dispose();
        }

        #endregion

        #region Helpers

        private int Detect(Mat image,
                           int resizedWidth,
                           int resizedHeight,
                           float scoreThreshold,
                           float nmsThreshold,
                           List<FaceInfo> faces)
        {
            if (image.IsEmpty)
            {
                Console.WriteLine("image is empty ,please check!");
                return -1;
            }

            this._ImageH = image.H;
            this._ImageW = image.W;

            this._ScaleW = (float)this._ImageW / resizedWidth;
            this._ScaleH = (float)this._ImageH / resizedHeight;

            using (Mat @in = new Mat())
            {
                //scale 
                this.DynamicScale(resizedWidth, resizedHeight);
                Ncnn.ResizeBilinear(image, @in, this._DW, this._DH);

                using (Extractor ex = this._Net.CreateExtractor())
                {
                    ex.Input("input.1", @in);

                    using (Mat heatMap = new Mat())
                    {
                        using (Mat scale = new Mat())
                        {
                            using (Mat offset = new Mat())
                            {
                                using (Mat landmarks = new Mat())
                                {
                                    ex.Extract("537", heatMap);
                                    ex.Extract("538", scale);
                                    ex.Extract("539", offset);
                                    ex.Extract("540", landmarks);

                                    this.Decode(heatMap, scale, offset, landmarks, faces, scoreThreshold, nmsThreshold);
                                    this.SquareBox(faces);
                                }
                            }
                        }
                    }
                }
            }

            return 0;
        }

        private void Decode(Mat heatMap,
                            Mat scale,
                            Mat offset,
                            Mat landmarks,
                            List<FaceInfo> faces,
                            float scoreThresh,
                            float nmsThresh)
        {
            int featH = heatMap.H;
            int featW = heatMap.W;
            int spacialSize = featW * featH;

            unsafe
            {
                float* heatMapData = (float*)(heatMap.Data);

                float* scale0 = (float*)(scale.Data);
                float* scale1 = scale0 + spacialSize;

                float* offset0 = (float*)(offset.Data);
                float* offset1 = offset0 + spacialSize;

                List<int> ids = new List<int>();
                this.GenIds(heatMapData, featH, featW, scoreThresh, ids);

                var facesTmp = new List<FaceInfo>();
                for (int i = 0; i < ids.Count / 2; i++)
                {
                    int idH = ids[2 * i];
                    int idW = ids[2 * i + 1];
                    int index = idH * featW + idW;

                    float s0 = (float)Math.Exp(scale0[index]) * 4;
                    float s1 = (float)Math.Exp(scale1[index]) * 4;
                    float o0 = offset0[index];
                    float o1 = offset1[index];

                    //std::cout << s0 << " " << s1 << " " << o0 << " " << o1 << std::endl;

                    float x1 = (float)((idW + o1 + 0.5) * 4 - s1 / 2 > 0.0f ? (idW + o1 + 0.5) * 4 - s1 / 2 : 0);
                    float y1 = (float)((idH + o0 + 0.5) * 4 - s0 / 2 > 0 ? (idH + o0 + 0.5) * 4 - s0 / 2 : 0);
                    x1 = x1 < this._DW ? x1 : this._DW;
                    y1 = y1 < this._DH ? y1 : this._DH;
                    float x2 = x1 + s1 < (float)this._DW ? x1 + s1 : this._DW;
                    float y2 = y1 + s0 < (float)this._DH ? y1 + s0 : this._DH;

                    //std::cout << X1 << " " << Y1 << " " << X2 << " " << Y2 << std::endl;

                    FaceInfo faceBox = new FaceInfo();
                    faceBox.X1 = x1;
                    faceBox.Y1 = y1;
                    faceBox.X2 = x2;
                    faceBox.Y2 = y2;
                    faceBox.Score = heatMapData[index];
                    faceBox.Area = (faceBox.X2 - faceBox.X1) * (faceBox.Y2 - faceBox.Y1);


                    float boxW = x2 - x1; //=s1?
                    float boxH = y2 - y1; //=s0?

                    //std::cout << facebox.X1 << " " << facebox.Y1 << " " << facebox.X2 << " " << facebox.Y2 << std::endl;
                    for (int j = 0; j < 5; j++)
                    {
                        float* xMap = (float*)landmarks.Data + (2 * j + 1) * spacialSize;
                        float* yMap = (float*)landmarks.Data + (2 * j) * spacialSize;
                        faceBox.Landmarks[2 * j] = x1 + xMap[index] * s1; //box_w;
                        faceBox.Landmarks[2 * j + 1] = y1 + yMap[index] * s0; // box_h;
                    }

                    facesTmp.Add(faceBox);
                }

                this.NonMaximumSuppression(facesTmp, faces, nmsThresh);
            }

            for (int k = 0; k < faces.Count; k++)
            {
                faces[k].X1 *= this._DScaleW * this._ScaleW;
                faces[k].Y1 *= this._DScaleH * this._ScaleH;
                faces[k].X2 *= this._DScaleW * this._ScaleW;
                faces[k].Y2 *= this._DScaleH * this._ScaleH;

                for (int kk = 0; kk < 5; kk++)
                {
                    faces[k].Landmarks[2 * kk] *= this._DScaleW * this._ScaleW;
                    faces[k].Landmarks[2 * kk + 1] *= this._DScaleH * this._ScaleH;
                }
            }
        }

        private void DynamicScale(float inW, float inH)
        {
            this._DH = (int)(Math.Ceiling(inH / 32) * 32);
            this._DW = (int)(Math.Ceiling(inW / 32) * 32);

            this._DScaleH = inH / this._DH;
            this._DScaleW = inW / this._DW;
        }

        private unsafe void GenIds(float* heatMap, int h, int w, float thresh, IList<int> ids)
        {
            if (heatMap == null)
            {
                Console.WriteLine($"{nameof(heatMap)} is nullptr,please check! ");
                return;
            }

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                    if (heatMap[i * w + j] > thresh)
                    {
                        ids.Add(i);
                        ids.Add(j);
                    }
            }
        }

        private void NonMaximumSuppression(List<FaceInfo> input, List<FaceInfo> output, float nmsThreshold = 0.3f, NonMaximumSuppressionMode type = NonMaximumSuppressionMode.Minimum)
        {
            if (!input.Any())
            {
                return;
            }

            input.Sort((f1, f2) => f1.Score.CompareTo(f2.Score));

            int nPick = 0;
            List<Tuple<float, int>> vScores = new List<Tuple<float, int>>();
            int numBoxes = input.Count;
            int[] vPick = new int[numBoxes];
            for (int i = 0; i < numBoxes; ++i)
            {
                vScores.Add(new Tuple<float, int>(input[i].Score, i));
            }

            while (vScores.Count > 0)
            {
                int last = vScores.Last().Item2;
                vPick[nPick] = last;
                nPick += 1;

                for (int index = 0; index < vScores.Count;)
                {
                    Tuple<float, int> it = vScores[index];
                    int itemIndex = it.Item2;

                    float maxX = Math.Max(input[itemIndex].X1, input[last].X1);
                    float maxY = Math.Max(input[itemIndex].Y1, input[last].Y1);
                    float minX = Math.Min(input[itemIndex].X2, input[last].X2);
                    float minY = Math.Min(input[itemIndex].Y2, input[last].Y2);
                    //maxX1 and maxY1 reuse 
                    maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
                    maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
                    //IOU reuse for the area of two bbox
                    float iou = maxX * maxY;
                    switch (type)
                    {
                        case NonMaximumSuppressionMode.Union:
                            iou = iou / (input[itemIndex].Area + input[last].Area - iou);
                            break;
                        case NonMaximumSuppressionMode.Minimum:
                            iou = iou / ((input[itemIndex].Area < input[last].Area) ? input[itemIndex].Area : input[last].Area);
                            break;
                    }

                    if (iou > nmsThreshold)
                    {
                        vScores.RemoveAt(index);
                    }
                    else
                    {
                        index++;
                    }
                }
            }

            Array.Resize(ref vPick, nPick);
            Resize(output, nPick);
            for (int i = 0; i < nPick; i++)
            {
                output[i] = input[vPick[i]];
            }
        }

        private void SquareBox(IList<FaceInfo> faces)
        {
            for (int i = 0; i < faces.Count; i++)
            {
                float w = faces[i].X2 - faces[i].X1;
                float h = faces[i].Y2 - faces[i].Y1;

                float maxSize = w < h ? h : w;
                float cenX = faces[i].X1 + w / 2;
                float cenY = faces[i].Y1 + h / 2;

                faces[i].X1 = cenX - maxSize / 2 > 0 ? cenX - maxSize / 2 : 0;
                faces[i].Y1 = cenY - maxSize / 2 > 0 ? cenY - maxSize / 2 : 0;
                faces[i].X2 = cenX + maxSize / 2 > this._ImageW - 1 ? this._ImageW - 1 : cenX + maxSize / 2;
                faces[i].Y2 = cenY + maxSize / 2 > this._ImageH - 1 ? this._ImageH - 1 : cenY + maxSize / 2;
            }
        }

        private static void Resize<T>(List<T> list, int size)
            where T : new()
        {
            int count = list.Count;
            if (size < count)
            {
                list.RemoveRange(size, count - size);
            }
            else
            {
                if (size > count)
                {
                    if (size > list.Capacity)
                    {
                        list.Capacity = size;
                    }
                    list.AddRange(Enumerable.Repeat(new T(), size - count));
                }
            }
        }

        public bool IsModelExists() => File.Exists(BinPath) && File.Exists(ParamPath);

        public bool TryInitialize(bool raiseError = false)
        {

            if (_centerFaceParameter != null)
            {
                return true;
            }

            if (!IsModelExists())
            {
                string message = $"Could not find the model of {DetectorName} detector";

                if (raiseError)
                {
                    throw new Exception(message);
                }
                // TaskLogger in future instead Console
                Console.WriteLine(message);
                return false;
            }
            return true;
        }



        #endregion

        #endregion

        public List<Face> DetectFaces(string imagePath)
        {
            List<Face> faces = new List<Face>() { };
            float tolerance = 0.5f;

            using (NcnnDotNet.OpenCV.Mat frame = NcnnDotNet.OpenCV.Cv2.ImRead(imagePath))
            {
                using (NcnnDotNet.Mat inMat = NcnnDotNet.Mat.FromPixels(frame.Data, NcnnDotNet.PixelType.Bgr2Rgb, frame.Cols, frame.Rows))
                {

                    FaceInfo[] faceInfos = _ultraFaceDetector.Detect(inMat, frame.Cols, frame.Rows, tolerance).ToArray();

                    foreach (FaceInfo detectedFace in faceInfos)
                    {
                        faces.Add(new Face((int)detectedFace.X1,
                                           (int)detectedFace.Y1,
                                           (int)detectedFace.X2 - (int)detectedFace.X1,
                                           (int)detectedFace.Y2 - (int)detectedFace.Y1,
                                           DrawPen));
                    }
                }
            }
            return faces;
        }
    }
}
