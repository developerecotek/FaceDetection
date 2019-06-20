using FaceRecognitionDotNet;
using Microsoft.Extensions.CommandLineUtils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace FaceDetection
{

    public class Program
    {

        #region Fields

        private static FaceRecognition _FaceRecognition;

        #endregion

        #region Methods


        ///Nếu thiếu lib DlibDotNet => vào nutget search và tải DlibDotNet
        ///https://github.com/takuya-takeuchi/FaceRecognitionDotNet
        private static void Main(string[] args)
        {
            var app = new CommandLineApplication(false);
            app.Name = nameof(FaceDetection);
            app.HelpOption("-h|--help");

            var directoryOption = app.Option("-d|--directory", "The directory path which includes image files", CommandOptionType.SingleValue);
            var cpuOption = app.Option("-c|--cpus", "The number of CPU cores to use in parallel. -1 means \"use all in system\"", CommandOptionType.SingleValue);
            var modelOption = app.Option("-m|--model", "Which face detection model to use. Options are \"hog\" or \"cnn\".", CommandOptionType.SingleValue);


            ///tải models tại: https://github.com/ageitgey/face_recognition_models/tree/master/face_recognition_models/models
            ///cho vào debug - nơi chạy program
            app.OnExecute(() =>
            {
                ///Các hàm khởi tạo môi trường
                var imageToCheck = "ImageLib";
                if (directoryOption.HasValue())
                    imageToCheck = directoryOption.Value();

                var strCpus = "1";
                if (cpuOption.HasValue())
                    strCpus = cpuOption.Value();

                var strModel = "Hog";
                if (modelOption.HasValue())
                    strModel = modelOption.Value();

                if (!Enum.TryParse<Model>(strModel, true, out var model))
                {
                    app.ShowHelp();
                    Console.WriteLine($"\n\tmodel: {strModel}");
                    return -1;
                }

                if (!int.TryParse(strCpus, out var cpus))
                {
                    app.ShowHelp();
                    Console.WriteLine($"\n\tcpus: {strCpus}");
                    return -1;
                }

                if (!Directory.Exists(imageToCheck))
                {
                    app.ShowHelp();
                    Console.WriteLine($"\n\tdirectory: {imageToCheck}");
                    return -1;
                }

                var directory = Path.GetFullPath("models");
                if (!Directory.Exists(directory))
                {
                    Console.WriteLine($"Please check whether model directory '{directory}' exists");
                    return -1;
                }

                _FaceRecognition = FaceRecognition.Create(directory);


                ////khoang vùng khuôn mặt
                ///input: file trong            debug/ImageLib
                ///output: ảnh trong forder     debug/output
                //if (Directory.Exists(imageToCheck))
                //    switch (cpus)
                //    {
                //        case 1:
                //            foreach (var imageFile in ImageFilesInFolder(imageToCheck))
                //            {
                //                Console.WriteLine($"{imageFile}...");
                //                TestImage(imageFile, model);
                //            }
                //            break;
                //        default:
                //            ProcessImagesInProcessPool(ImageFilesInFolder(imageToCheck), cpus, model);
                //            break;
                //    }
                //else
                //    TestImage(imageToCheck, model);

                ///So sánh tắt cả các khuôn mặt có trong ảnh với mới mặt cho trước.
                //string face = "ImageLib/check.jpg";
                //string face = "ImageLib/check2.jpg";
                string face = "ImageLib/buffertest.jpg";
                string dir = "ImageLib";
                CompareImage(dir, face);

                return 0;
            });

            app.Execute(args);
            var output = new DirectoryInfo("output").FullName;
            Console.WriteLine("Press enter to exit...");
            Console.ReadKey();
        }

        #region Helpers

        private static IEnumerable<string> ImageFilesInFolder(string folder)
        {
            return Directory.GetFiles(folder)
                            .Where(s => Regex.IsMatch(Path.GetExtension(s), "(jpg|jpeg|png)$", RegexOptions.Compiled));
        }

        private static void PrintResult(string filename, Location location)
        {
            System.Drawing.Image image = new Bitmap(filename);
            Graphics graph = Graphics.FromImage(image);
            Pen pen = new Pen(Brushes.Red);
            Rectangle rect = new Rectangle(new System.Drawing.Point(location.Left, location.Top), new Size(location.Right - location.Left, location.Bottom - location.Top));
            Console.WriteLine($"{location.Left}\t{location.Top}\t{location.Right}\t{location.Bottom}");
            graph.DrawRectangle(pen, rect);
            var save = $"output/{location.Left}L_{location.Top}T_{location.Right}R_{location.Bottom}B_{Path.GetFileName(filename)}";
            string outputFileName = save;
            using (MemoryStream memory = new MemoryStream())
            {
                using (FileStream fs = new FileStream(outputFileName, FileMode.Create, FileAccess.ReadWrite))
                {
                    image.Save(memory, ImageFormat.Jpeg);
                    byte[] bytes = memory.ToArray();
                    fs.Write(bytes, 0, bytes.Length);
                }
            }
            Console.WriteLine($">> save in {Path.GetFullPath(save)}");
        }

        private static void ProcessImagesInProcessPool(IEnumerable<string> imagesToCheck, int numberOfCpus, Model model)
        {
            if (numberOfCpus == -1)
                numberOfCpus = Environment.ProcessorCount;

            var files = imagesToCheck.ToArray();
            var functionParameters = files.Select(s => new Tuple<string, Model>(s, model)).ToArray();

            var total = functionParameters.Length;
            var option = new ParallelOptions
            {
                MaxDegreeOfParallelism = numberOfCpus
            };

            Parallel.For(0, total, option, i =>
            {
                var t = functionParameters[i];
                TestImage(t.Item1, t.Item2);
            });
        }

        public static void TestImage(string imageToCheck, Model model)
        {
            using (var unknownImage = FaceRecognition.LoadImageFile(imageToCheck))
            {
                var faceLocations = _FaceRecognition.FaceLocations(unknownImage, 0, model).ToArray();

                foreach (var faceLocation in faceLocations)
                {
                    PrintResult(imageToCheck, faceLocation);
                }
            }
        }

        public static void CompareImage(string dir, string face)
        {
            var files = Directory.GetFiles(dir);
            using (var knowface = FaceRecognition.LoadImageFile(face))
            {
                var knowface_endcode = _FaceRecognition.FaceEncodings(knowface);
                foreach (var item in files)
                {
                    using (var unknownImage = FaceRecognition.LoadImageFile(item))
                    {
                        var encoding = _FaceRecognition.FaceEncodings(unknownImage);
                        foreach (var enc in encoding)
                        {
                            foreach (var e in FaceRecognition.CompareFaces(knowface_endcode, enc))
                            {
                                Console.WriteLine($"{e} - {item}");
                            }
                        }
                    }
                }
            }
        }

        #endregion

        #endregion

    }

}
