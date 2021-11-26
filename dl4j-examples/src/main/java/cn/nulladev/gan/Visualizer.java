package cn.nulladev.gan;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Visualizer {

    private double imageScale;
    private List<INDArray> digits;
    private String title;
    private int gridWidth;

    public Visualizer(double imageScale, List<INDArray> digits, String title) {
        this(imageScale, digits, title, 8);
    }

    public Visualizer(double imageScale, List<INDArray> digits, String title, int gridWidth) {
        this.imageScale = imageScale;
        this.digits = digits;
        this.title = title;
        this.gridWidth = gridWidth;
    }

    public void visualize() {
        JFrame frame = new JFrame();
        frame.setTitle(title);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(0, gridWidth));

        List<JLabel> list = getComponents();
        for (JLabel image : list) {
            panel.add(image);
        }

        frame.add(panel);
        frame.setVisible(true);
        frame.pack();
    }

    public List<JLabel> getComponents() {
        List<JLabel> images = new ArrayList<>();
        for (INDArray arr : digits) {
            BufferedImage bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
            for (int i = 0; i < 784; i++) {
                bi.getRaster().setSample(i % 28, i / 28, 0, (int) (255 * arr.getDouble(i)));
            }
            ImageIcon orig = new ImageIcon(bi);
            Image imageScaled = orig.getImage().getScaledInstance((int) (imageScale * 28), (int) (imageScale * 28),
                    Image.SCALE_DEFAULT);
            ImageIcon scaled = new ImageIcon(imageScaled);
            images.add(new JLabel(scaled));
        }
        return images;
    }

    public static void showBest(ComputationGraph model) throws Exception{
        DataSetIterator train = new MnistDataSetIterator(32, true, 19260817);
        INDArray real = train.next().getFeatures();
        Map<String, INDArray> map = model.feedForward(
                new INDArray[] {Nd4j.rand(new long[] {64,10}, new NormalDistribution()), real}, false
        );
        INDArray indArray = map.get("gen_l3");
        List<INDArray> list = new ArrayList<>();
        for (int j = 0; j < indArray.size(0); j++) {
            list.add(indArray.getRow(j));
        }

        Visualizer bestVisualizer = new Visualizer(1, list, "GAN");
        bestVisualizer.visualize();
    }

}
