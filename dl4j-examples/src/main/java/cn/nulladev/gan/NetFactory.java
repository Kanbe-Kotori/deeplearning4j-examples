package cn.nulladev.gan;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.*;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.StackVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NetFactory {

    public static final int MNIST_WIDTH = 28;
    public static final int MNIST_HEIGHT = 28;
    public static final int SIZE = MNIST_WIDTH * MNIST_HEIGHT;

    public static final double LEARNING_RATE = 0.01D;
    public static final double LEARNING_RATE2 = 0.00001D;

    public static final double MAX_W = 0.01D;

    public static GraphBuilder createBuilder() {
        GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(19260817L)
                .updater(new Sgd(LEARNING_RATE))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .backpropType(BackpropType.Standard)
                .addInputs("input_1", "input_2")
                .addLayer("gen_l1",
                        new DenseLayer.Builder()
                                .nIn(10)
                                .nOut(128)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER)
                                .build(),
                        "input_1")
                .addLayer("gen_l2",
                        new DenseLayer.Builder()
                                .nIn(128)
                                .nOut(512)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER)
                                .build(),
                        "gen_l1")
                .addLayer("gen_l3",
                        new DenseLayer.Builder()
                                .nIn(512)
                                .nOut(SIZE)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER)
                                .build(),
                        "gen_l2")
                .addVertex("stack", new StackVertex(), "input_2", "gen_l3")
                .addLayer("dis_l1",
                        new DenseLayer.Builder()
                                .nIn(SIZE)
                                .nOut(256)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER)
                                .build(),
                        "stack")
                .addLayer("dis_l2",
                        new DenseLayer.Builder()
                                .nIn(256)
                                .nOut(128)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER)
                                .build(),
                        "dis_l1")
                .addLayer("dis_l3",
                        new DenseLayer.Builder()
                                .nIn(128)
                                .nOut(128)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER)
                                .build(),
                        "dis_l2")
                .addLayer("out",
                        new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                                .nIn(128)
                                .nOut(1)
                                .activation(Activation.SIGMOID)
                                .build(),
                        "dis_l3")
                .setOutputs("out");
        return builder;
    }

    public static GraphBuilder createBuilder2() {
        GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(19260817L)
                .updater(new Sgd(LEARNING_RATE2))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .backpropType(BackpropType.Standard)
                .addInputs("input_1", "input_2")
                .addLayer("gen_l1",
                        new DenseLayer.Builder()
                                .nIn(10)
                                .nOut(128)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER)
                                .build(),
                        "input_1")
                .addLayer("gen_l2",
                        new DenseLayer.Builder()
                                .nIn(128)
                                .nOut(512)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER)
                                .build(),
                        "gen_l1")
                .addLayer("gen_l3",
                        new DenseLayer.Builder()
                                .nIn(512)
                                .nOut(SIZE)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER)
                                .build(),
                        "gen_l2")
                .addVertex("stack", new StackVertex(), "input_2", "gen_l3")
                .addLayer("dis_l1",
                        new DenseLayer.Builder()
                                .nIn(SIZE)
                                .nOut(256)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER)
                                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                .gradientNormalizationThreshold(MAX_W)
                                .build(),
                        "stack")
                .addLayer("dis_l2",
                        new DenseLayer.Builder()
                                .nIn(256)
                                .nOut(128)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER)
                                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                .gradientNormalizationThreshold(MAX_W)
                                .build(),
                        "dis_l1")
                .addLayer("dis_l3",
                        new DenseLayer.Builder()
                                .nIn(128)
                                .nOut(128)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER)
                                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                .gradientNormalizationThreshold(MAX_W)
                                .build(),
                        "dis_l2")
                .addLayer("out",
                        new OutputLayer.Builder(LossFunctions.LossFunction.WASSERSTEIN)
                                .nIn(128)
                                .nOut(1)
                                .activation(Activation.IDENTITY)
                                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                .gradientNormalizationThreshold(MAX_W)
                                .build(),
                        "dis_l3")
                .setOutputs("out");
        return builder;
    }

    public static void set_lr_to_dis(ComputationGraph net) {
        net.setLearningRate("gen_l1", 0);
        net.setLearningRate("gen_l2", 0);
        net.setLearningRate("gen_l3", 0);
        net.setLearningRate("dis_l1", LEARNING_RATE);
        net.setLearningRate("dis_l2", LEARNING_RATE);
        net.setLearningRate("dis_l3", LEARNING_RATE);
        net.setLearningRate("out", LEARNING_RATE);
    }

    public static void set_lr_to_gen(ComputationGraph net) {
        net.setLearningRate("gen_l1", LEARNING_RATE);
        net.setLearningRate("gen_l2", LEARNING_RATE);
        net.setLearningRate("gen_l3", LEARNING_RATE);
        net.setLearningRate("dis_l1", 0);
        net.setLearningRate("dis_l2", 0);
        net.setLearningRate("dis_l3", 0);
        net.setLearningRate("out", 0);
    }

    public static void set_lr_to_dis2(ComputationGraph net) {
        net.setLearningRate("gen_l1", 0);
        net.setLearningRate("gen_l2", 0);
        net.setLearningRate("gen_l3", 0);
        net.setLearningRate("dis_l1", LEARNING_RATE2);
        net.setLearningRate("dis_l2", LEARNING_RATE2);
        net.setLearningRate("dis_l3", LEARNING_RATE2);
        net.setLearningRate("out", LEARNING_RATE2);
    }

    public static void set_lr_to_gen2(ComputationGraph net) {
        net.setLearningRate("gen_l1", LEARNING_RATE2);
        net.setLearningRate("gen_l2", LEARNING_RATE2);
        net.setLearningRate("gen_l3", LEARNING_RATE2);
        net.setLearningRate("dis_l1", 0);
        net.setLearningRate("dis_l2", 0);
        net.setLearningRate("dis_l3", 0);
        net.setLearningRate("out", 0);
    }

}
