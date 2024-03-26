package com.hyfly.unit01;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import cn.hutool.core.io.FileUtil;
import lombok.extern.slf4j.Slf4j;
import tech.tablesaw.api.BooleanColumn;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;

import java.io.File;
import java.io.FileWriter;
import java.util.List;

@Slf4j
public class Test03 {

    public static void main(String[] args) throws Exception {

        File mkdir = FileUtil.mkdir("./data");

        if (mkdir.exists()) {
            String dataFile = "./data/house_tiny.csv";

            File file = FileUtil.newFile(dataFile);

            if (file.exists()) {
                // Write to file
                try (FileWriter writer = new FileWriter(dataFile)) {
                    writer.write("NumRooms,Alley,Price\n"); // Column names
                    writer.write("NA,Pave,127500\n");  // Each row represents a data example
                    writer.write("2,NA,106000\n");
                    writer.write("4,NA,178100\n");
                    writer.write("NA,NA,140000\n");
                }

                Table data = Table.read().file(dataFile);

                log.info(data.toString());

                // 处理缺失的数据
                Table inputs = data.create(data.columns());
                inputs.removeColumns("Price");
                Table outputs = data.select("Price");

                log.info(outputs.toString());

                Column col = inputs.column("NumRooms");
                col.set(col.isMissing(), (int) inputs.nCol("NumRooms").mean());

                log.info(inputs.toString());

                StringColumn col2 = (StringColumn) inputs.column("Alley");
                List<BooleanColumn> dummies = col2.getDummies();
                inputs.removeColumns(col2);
                inputs.addColumns(DoubleColumn.create("Alley_Pave", dummies.get(0).asDoubleArray()),
                        DoubleColumn.create("Alley_nan", dummies.get(1).asDoubleArray())
                );

                log.info(inputs.toString());

                try (NDManager nd = NDManager.newBaseManager()) {
                    NDArray x = nd.create(inputs.as().doubleMatrix());
                    NDArray y = nd.create(outputs.as().intMatrix());

                    log.info(x.toDebugString(true));
                    log.info(y.toDebugString(true));
                }
            } else {
                log.info("Create file failed");
            }
        } else {
            log.info("Create directory failed");
        }
    }


}
