package cn.nulladev.test;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NetFactory {

    public static MultiLayerConfiguration CWRUANN() {
        MultiLayerConfiguration builder = new NeuralNetConfiguration.Builder()
                .seed(19260817L)
                .updater(new Sgd(0.01))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(512).nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(128).nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(32).nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(10).nOut(10).build())
                .build();
        return builder;
    }

    public static MultiLayerConfiguration CWRUAE() {
        MultiLayerConfiguration builder = new NeuralNetConfiguration.Builder()
                .seed(19260817L)
                .updater(new Sgd(0.0001))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(512).nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(128).nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(16).nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(128).nOut(512).build())
                .build();
        return builder;
    }

    public static MultiLayerConfiguration CWRUConf2() {
        MultiLayerConfiguration builder = new NeuralNetConfiguration.Builder()
                .seed(19260817L)
                .updater(new Sgd(0.01))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new Convolution1DLayer.Builder()
                        .kernelSize(64)
                        .stride(16)
                        .activation(Activation.LEAKYRELU)
                        .nOut(1)
                        .build())
                .build();
        return builder;
    }
}
