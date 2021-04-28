using System.Drawing;

namespace FaceDetector
{
    public class Face
    {
        public Face(int X1, int Y1, int X2, int Y2, Pen pen)
        {
            this.X1 = X1;
            this.Y1 = Y1;
            this.X2 = X2;
            this.Y2 = Y2;
            Pen = pen;
        }

        public int X1 { get; private set; }
        public int Y1 { get; private set; }
        public int X2 { get; private set; }
        public int Y2 { get; private set; }
        public Pen Pen { get; private set; }

        public override string ToString()
        {
            return $"[{X1}, {Y1}, {X2}, {Y2}]";
        }
    }
}
