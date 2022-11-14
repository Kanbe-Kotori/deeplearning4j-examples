package cn.nulladev.test;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.*;
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

    public static MultiLayerConfiguration CWRUCNN() {
        MultiLayerConfiguration builder = new NeuralNetConfiguration.Builder()
                .seed(19260817L)
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .convolutionMode(ConvolutionMode.Same)
                .list()
                .layer(new Convolution1DLayer.Builder()
                        .kernelSize(64)
                        .padding(0)
                        .stride(16)
                        .activation(Activation.RELU).nOut(16).build())
                .layer(new Subsampling1DLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2)
                        .stride(2)
                        .build())
                .layer(new Convolution1DLayer.Builder()
                        .kernelSize(3)
                        .padding(0)
                        .stride(1)
                        .activation(Activation.RELU).nOut(32).build())
                .layer(new Subsampling1DLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2)
                        .stride(2)
                        .build())
                .layer(new Convolution1DLayer.Builder()
                        .kernelSize(3)
                        .padding(0)
                        .stride(1)
                        .activation(Activation.RELU).nOut(64).build())
                .layer(new Subsampling1DLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2)
                        .stride(2)
                        .build())
                .layer(new Convolution1DLayer.Builder()
                        .kernelSize(3)
                        .stride(1)
                        .activation(Activation.RELU).nOut(64).build())
                .layer(new Subsampling1DLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2)
                        .stride(2)
                        .build())
                .layer(new DenseLayer.Builder().nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(100).nOut(10).build())
                .setInputType(InputType.feedForward(2048))
                .build();
        return builder;
    }

    public static MultiLayerConfiguration CWRUAE() {
        MultiLayerConfiguration builder = new NeuralNetConfiguration.Builder()
                .seed(19260817L)
                .updater(new Sgd(0.01))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(512).nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(256).nOut(512).build())
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
