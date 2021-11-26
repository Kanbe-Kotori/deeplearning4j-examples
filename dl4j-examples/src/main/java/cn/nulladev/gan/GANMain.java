package cn.nulladev.gan;

import org.deeplearning4j.nn.graph.ComputationGraph;

import java.io.File;

public class GANMain {

    public static void main(String[] Args) {
        try {
            GANTrainer.train2();
            ComputationGraph restored = ComputationGraph.load(new File("E:/Temp/wgan-10000.zip"), true);
            Visualizer.showBest(restored);
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

}
