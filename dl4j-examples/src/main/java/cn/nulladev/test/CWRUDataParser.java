package cn.nulladev.test;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

import java.io.File;

public class CWRUDataParser {

    public static void parse() throws Exception {
        var path = "src\\main\\java\\cn\\nulladev\\test\\dataset";
        for (String fname : new File(path).list()) {
            //System.out.println(fname);
            CWRUData d = new CWRUData();
            d.name = fname;
            CWRUDataManager.dataList.add(d);

            MatFileReader reader = new MatFileReader(path + "\\" + fname);
            var content = reader.getContent();

            if (fname.contains("_B")) {
                d.err_type = "Ball";
            } else if (fname.contains("_IR")) {
                d.err_type = "IR";
            } else if (fname.contains("_OR")) {
                d.err_type = "OR";
            } else {
                d.err_type = "Normal";
            }

            if (fname.contains("028")) {
                d.depth = 28;
            } else if (fname.contains("021")) {
                d.depth = 21;
            } else if (fname.contains("014")) {
                d.depth = 14;
            } else if (fname.contains("007")) {
                d.depth = 7;
            }

            if (fname.contains("_0_")) {
                d.load = 0;
            } else if (fname.contains("_1_")) {
                d.load = 1;
            } else if (fname.contains("_2_")) {
                d.load = 2;
            } else if (fname.contains("_3_")) {
                d.load = 3;
            }

            if (fname.contains("@3")) {
                d.pos = 3;
            } else if (fname.contains("@6")) {
                d.pos = 6;
            } else if (fname.contains("@12")) {
                d.pos = 12;
            }

            for (String key : content.keySet()) {
                var value = content.get(key);
                if (key.contains("DE")) {
                    d.DE = d.err_type.equals("Normal")?toDoubleArray4x(value): toDoubleArray(value);
                    //System.out.println(d.DE.length);
                } else if (key.contains("FE")) {
                    d.FE = d.err_type.equals("Normal")?toDoubleArray4x(value): toDoubleArray(value);
                    //System.out.println(d.FE.length);
                } else if (key.contains("BA")) {
                    d.BA = d.err_type.equals("Normal")?toDoubleArray4x(value): toDoubleArray(value);
                    //System.out.println(d.BA.length);
                } else if (key.contains("RPM")){
                    d.rpm = Double.valueOf(value.contentToString().split("=")[1]);
                    //System.out.println(d.rpm);
                }
            }
        }
    }

    public static double[] toDoubleArray(MLArray ma) {
        MLDouble md = (MLDouble) ma;
        int m = md.getM();
        double[] data = new double[m];
        for (int i = 0; i < m; i++) {
            data[i] = md.get(i, 0);
        }
        return data;
    }

    public static double[] toDoubleArray4x(MLArray ma) {
        MLDouble md = (MLDouble) ma;
        int m = md.getM();
        double[] data = new double[m/4];
        for (int i = 0; i < m/4; i++) {
            data[i] = (md.get(4*i, 0)+md.get(4*i+1, 0)+md.get(4*i+2, 0)+md.get(4*i+3, 0))/4;
        }
        return data;
    }
}
