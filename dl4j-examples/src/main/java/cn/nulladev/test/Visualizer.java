package cn.nulladev.test;

import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Iterator;

public class Visualizer {

    public static void INDArray2IMG(INDArray array, String name) throws Exception {
        BufferedImage bi = new BufferedImage(512, 128, BufferedImage.TYPE_INT_ARGB);
        int prevH = 63;
        for(int i = 0; i < 512; i++) {
            for(int j = 0; j < 128; j++) {
                bi.setRGB(i, j, 0xFFFFFFFF);
            }
            int h = (int) (64 * array.getDouble(i) + 64);
            h = Math.min(h,127);
            h = Math.max(h,0);
            for(int j = Math.min(prevH, h); j < Math.max(prevH, h); j++) {
                bi.setRGB(i, j, 0xFFFF0000);
            }
            prevH = h;
        }
        writeIMG(bi, name);
    }

    public static void writeIMG(BufferedImage bi, String name) throws Exception {
        Iterator<ImageWriter> it = ImageIO.getImageWritersByFormatName("png");
        ImageWriter writer = it.next();
        ImageOutputStream ios = ImageIO.createImageOutputStream(new File(name));
        writer.setOutput(ios);
        writer.write(bi);
        bi.flush();
        ios.flush();
    }

}
