package cn.nulladev.gan;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

public class GANTrainer {

    public static final int BATCH_SIZE = 32;

    public static void train() throws Exception{
        //ComputationGraph net = new ComputationGraph(NetFactory.createBuilder().build());
        //net.init();
        //System.out.println(net.summary());
        ComputationGraph net = ComputationGraph.load(new File("E:/Temp/gan-110000.zip"), true);
        net.setListeners(new ScoreIterationListener(100));

        DataSetIterator mnist = new MnistDataSetIterator(BATCH_SIZE, true, 19260817);
        INDArray label_gen = Nd4j.ones(2 * BATCH_SIZE, 1); //All true
        INDArray label_dis = Nd4j.vstack(Nd4j.ones(BATCH_SIZE, 1), Nd4j.zeros(BATCH_SIZE, 1));  //First half all true, second half all false

        for (int i=1; i<=90000; i++) {
            if (!mnist.hasNext()) {
                mnist.reset();
            }
            INDArray real = mnist.next().getFeatures();
            INDArray rand = Nd4j.rand(new long[]{BATCH_SIZE, 10}, new NormalDistribution());
            MultiDataSet dataSetD = new MultiDataSet(new INDArray[]{rand,real}, new INDArray[]{label_dis});
            for (int j=0; j<10; j++) {
                NetFactory.set_lr_to_dis(net);
                net.fit(dataSetD);
            }
            rand = Nd4j.rand(new long[]{BATCH_SIZE, 10}, new NormalDistribution());
            MultiDataSet dataSetG = new MultiDataSet(new INDArray[] {rand,real}, new INDArray[]{label_gen});
            NetFactory.set_lr_to_gen(net);
            net.fit(dataSetG);
            if (i%10000==0) {
                net.save(new File("E:/Temp/gan-"+(i+110000)+".zip"), true);
            }
        }
    }

    public static void train2() throws Exception{
        ComputationGraph net = new ComputationGraph(NetFactory.createBuilder2().build());
        net.init();
        //System.out.println(net.summary());
        //ComputationGraph net = ComputationGraph.load(new File("E:/Temp/wgan-50000.zip"), true);
        net.setListeners(new ScoreIterationListener(100));

        DataSetIterator mnist = new MnistDataSetIterator(BATCH_SIZE, true, 19260817);
        INDArray label_gen = Nd4j.ones(2 * BATCH_SIZE, 1); //All true
        INDArray label_dis = Nd4j.vstack(Nd4j.ones(BATCH_SIZE, 1), Nd4j.ones(BATCH_SIZE, 1).muli(-1));  //First half all true, second half all false

        for (int i=1; i<=10000; i++) {
            if (!mnist.hasNext()) {
                mnist.reset();
            }
            INDArray real = mnist.next().getFeatures();
            INDArray rand = Nd4j.rand(new long[]{BATCH_SIZE, 10}, new NormalDistribution());
            MultiDataSet dataSetD = new MultiDataSet(new INDArray[]{rand,real}, new INDArray[]{label_dis});
            for (int j=0; j<10; j++) {
                NetFactory.set_lr_to_dis2(net);
                net.fit(dataSetD);
            }
            rand = Nd4j.rand(new long[]{BATCH_SIZE, 10}, new NormalDistribution());
            MultiDataSet dataSetG = new MultiDataSet(new INDArray[] {rand,real}, new INDArray[]{label_gen});
            NetFactory.set_lr_to_gen2(net);
            net.fit(dataSetG);
            if (i%10000==0) {
                net.save(new File("E:/Temp/wgan-"+(i)+".zip"), true);
            }
        }
    }

}
