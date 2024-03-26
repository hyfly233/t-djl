package com.hyfly;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class test02 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.create(new float[]{1f, 2f, 4f, 8f});
            NDArray y = manager.create(new float[]{2f, 2f, 2f, 2f});

            NDArray add = x.add(y);
            log.info("x + y = " + add.toDebugString(true));

            NDArray sub = x.sub(y);
            log.info("x - y = " + sub.toDebugString(true));

            NDArray mul = x.mul(y);
            log.info("x * y = " + mul.toDebugString(true));

            NDArray div = x.div(y);
            log.info("x / y = " + div.toDebugString(true));

            NDArray power = x.pow(y);
            log.info("x ^ y = " + power.toDebugString(true));

            NDArray mod = x.mod(y);
            log.info("x % y = " + mod.toDebugString(true));

            NDArray eq = x.eq(y);
            log.info("x == y = " + eq.toDebugString(true));

            NDArray neq = x.neq(y);
            log.info("x != y = " + neq.toDebugString(true));

            NDArray gt = x.gt(y);
            log.info("x > y = " + gt.toDebugString(true));

            NDArray gte = x.gte(y);
            log.info("x >= y = " + gte.toDebugString(true));

            NDArray lt = x.lt(y);
            log.info("x < y = " + lt.toDebugString(true));

            NDArray lte = x.lte(y);
            log.info("x <= y = " + lte.toDebugString(true));

            NDArray and = x.logicalAnd(y);
            log.info("x && y = " + and.toDebugString(true));

            NDArray or = x.logicalOr(y);
            log.info("x || y = " + or.toDebugString(true));

            NDArray not = x.logicalNot();
            log.info("!x = " + not.toDebugString(true));

            NDArray sin = x.sin();
            log.info("sin(x) = " + sin.toDebugString(true));

            NDArray cos = x.cos();
            log.info("cos(x) = " + cos.toDebugString(true));

            NDArray tan = x.tan();
            log.info("tan(x) = " + tan.toDebugString(true));

            NDArray asin = x.asin();
            log.info("asin(x) = " + asin.toDebugString(true));

            NDArray acos = x.acos();
            log.info("acos(x) = " + acos.toDebugString(true));

        }
    }
}
