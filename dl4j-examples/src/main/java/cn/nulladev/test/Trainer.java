package cn.nulladev.test;

import org.deeplearning4j.core.storage.StatsStorageRouter;
import org.deeplearning4j.core.storage.impl.RemoteUIStatsStorageRouter;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.examples.quickstart.modeling.feedforward.classification.IrisClassifier;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Trainer {

    private static Logger log = LoggerFactory.getLogger(IrisClassifier.class);

    public static void ANN(DataSet train, DataSet test) throws Exception {

        MultiLayerNetwork model = new MultiLayerNetwork(NetFactory.CWRUANN());
        model.init();

        UIServer server = UIServer.getInstance();
        server.enableRemoteListener();
        StatsStorageRouter remoteUIRouter = new RemoteUIStatsStorageRouter("http://localhost:9000");
        model.setListeners(new StatsListener(remoteUIRouter));

        DataSetIterator iterator = getIter(train, 20);

        for (int x = 0; x< 10000; x++) {
            if (!iterator.hasNext()) {
                iterator = getIter(train, 20);
            }
            model.fit(iterator);
            if (x % 10 == 0) System.out.println(x);
            if (x % 100 == 0) {
                model.save(new File("E:/Temp/aan-" + x + ".zip"), true);
                Evaluation eval = new Evaluation(10);
                INDArray output = model.output(test.getFeatures());
                eval.eval(test.getLabels(), output);
                log.info(eval.stats());
            }
        }
    }

    public static void ANNResult(DataSet train, DataSet test) throws Exception {
        MultiLayerNetwork model = MultiLayerNetwork.load(new File("E:/Temp/aan-" + 1000 + ".zip"), true);
        Evaluation eval = new Evaluation(10);
        INDArray output = model.output(test.getFeatures());
        eval.eval(test.getLabels(), output);
        log.info(eval.stats());
    }

    public static void AE(DataSet train, DataSet test) throws Exception {
        MultiLayerNetwork model = new MultiLayerNetwork(NetFactory.CWRUAE());
        model.init();

        UIServer server = UIServer.getInstance();
        server.enableRemoteListener();
        StatsStorageRouter remoteUIRouter = new RemoteUIStatsStorageRouter("http://localhost:9000");
        model.setListeners(new StatsListener(remoteUIRouter));

        DataSetIterator iterator = getIter(train, 1);

        for (int x = 0; x< 10000; x++) {
            if (!iterator.hasNext()) {
                iterator = getIter(train, 1);
            }
            model.fit(iterator);
            if (x%10==0)System.out.println(x);
            if (x%100==0) model.save(new File("E:/Temp/ae-"+x+".zip"), true);

            INDArray output = model.output(test.getFeatures());
            Visualizer.INDArray2IMG(output, "E:/Temp/vis-"+x+".png");
        }

    }

    private static DataSetIterator getIter(final DataSet set, final int batchSize) {
        final List<DataSet> list = set.asList();
        Collections.shuffle(list, new Random());
        return new ListDataSetIterator(list,batchSize);
    }
}
