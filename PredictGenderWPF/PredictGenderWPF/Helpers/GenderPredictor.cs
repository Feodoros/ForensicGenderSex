using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PredictGenderWPF
{
    public class GenderPredictor
    {
        private InferenceSession _session;

        public GenderPredictor()
        {
            string projectDirectory = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.Parent.FullName;
            string modelFilePath = Path.Combine(projectDirectory, "Models", "efnet_b0.onnx");
            _session = new InferenceSession(modelFilePath);
        }

        public string Predict(string imagePath)
        {
            // Read image
            using (Image<Rgb24> image = Image.Load<Rgb24>(imagePath))
            {
                // Resize image
                image.Mutate(x =>
                {
                    x.Resize(new ResizeOptions
                    {
                        Size = new Size(150, 150),
                        Mode = ResizeMode.Crop
                    });
                });

                // Preprocess image
                Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, 150, 150 });
                for (int y = 0; y < image.Height; y++)
                {
                    Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
                    for (int x = 0; x < image.Width; x++)
                    {
                        input[0, 0, y, x] = (pixelSpan[x].R / 255f);
                        input[0, 1, y, x] = (pixelSpan[x].G / 255f);
                        input[0, 2, y, x] = (pixelSpan[x].B / 255f);
                    }
                }
                
                // Setup inputs
                var inputs = new List<NamedOnnxValue>
                {
                NamedOnnxValue.CreateFromTensor("input.1", input)
                };

                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session?.Run(inputs);
                float output = results.First().AsEnumerable<float>().First();
                return (output >= 0.5) ? "Male" : "Female";
            }
        }
    }
}
