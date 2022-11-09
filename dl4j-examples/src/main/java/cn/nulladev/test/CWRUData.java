package cn.nulladev.test;

import java.util.ArrayList;
import java.util.List;

public class CWRUData {
    public String name;
    public String err_type;
    public int depth;
    public int pos = 0;
    public int load;

    public double[] DE;
    public double[] FE;
    public double[] BA;

    public double rpm;

    public List<CWRUBlock> blocks = new ArrayList();

    public void cut(int size, int num) {
        for (int current = 0; current < DE.length-size ;current += DE.length/num) {
            double[] blockDE = new double[size];
            for (int i = 0; i < size; i++) {
                blockDE[i] = DE[i + current];
            }
            CWRUBlock block = new CWRUBlock(size, this);
            block.DE = blockDE;
            blocks.add(block);
        }
    }

    public int type() {
        if (err_type.equals("Ball")) {
            return depth / 7;
        }
        if (err_type.equals("IR")) {
            return 3 + depth / 7;
        }
        if (err_type.equals("OR")) {
            return 6 + depth / 7;
        }
        return 0;
    }

    public void print() {
        System.out.println("length:" + DE.length);
        System.out.println("block num:" + blocks.size());
    }
}
