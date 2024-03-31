package com.hyfly.unit13;

import ai.djl.ndarray.NDManager;
import ai.djl.util.Pair;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Stream;

@Slf4j
public class Test1405 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            String[] symbols = {
                    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
                    "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "_", "[UNK]"
            };

            HashMap<String, Integer> rawTokenFreqs = new HashMap<>();
            rawTokenFreqs.put("fast_", 4);
            rawTokenFreqs.put("faster_", 3);
            rawTokenFreqs.put("tall_", 5);
            rawTokenFreqs.put("taller_", 4);

            HashMap<String, Integer> tokenFreqs = new HashMap<>();
            for (Map.Entry<String, Integer> e : rawTokenFreqs.entrySet()) {
                String token = e.getKey();
                tokenFreqs.put(String.join(" ", token.split("")), rawTokenFreqs.get(token));
            }

            System.out.println(tokenFreqs);

            int numMerges = 10;
            for (int i = 0; i < numMerges; i++) {
                Pair<String, String> maxFreqPair = getMaxFreqPair(tokenFreqs);
                Pair<HashMap<String, Integer>, String[]> pair =
                        mergeSymbols(maxFreqPair, tokenFreqs);
                tokenFreqs = pair.getKey();
                symbols =
                        Stream.concat(Arrays.stream(symbols), Arrays.stream(pair.getValue()))
                                .toArray(String[]::new);
                System.out.println(
                        "合并 #"
                                + (i + 1)
                                + ": ("
                                + maxFreqPair.getKey()
                                + ", "
                                + maxFreqPair.getValue()
                                + ")");
            }

            System.out.println(Arrays.toString(symbols));

            System.out.println(tokenFreqs.keySet());

            String[] tokens = new String[]{"tallest_", "fatter_"};
            System.out.println(segmentBPE(tokens, symbols));
        }
    }

    public static Pair<String, String> getMaxFreqPair(HashMap<String, Integer> tokenFreqs) {
        HashMap<Pair<String, String>, Integer> pairs = new HashMap<>();
        for (Map.Entry<String, Integer> e : tokenFreqs.entrySet()) {
            // Key of 'pairs' is a tuple of two consecutive symbols
            String token = e.getKey();
            Integer freq = e.getValue();
            String[] symbols = token.split(" ");
            for (int i = 0; i < symbols.length - 1; i++) {
                pairs.put(
                        new Pair<>(symbols[i], symbols[i + 1]),
                        pairs.getOrDefault(new Pair<>(symbols[i], symbols[i + 1]), 0) + freq);
            }
        }
        int max = 0; // Key of `pairs` with the max value
        Pair<String, String> maxFreqPair = null;
        for (Map.Entry<Pair<String, String>, Integer> pair : pairs.entrySet()) {
            if (max < pair.getValue()) {
                max = pair.getValue();
                maxFreqPair = pair.getKey();
            }
        }
        return maxFreqPair;
    }

    public static Pair<HashMap<String, Integer>, String[]> mergeSymbols(
            Pair<String, String> maxFreqPair, HashMap<String, Integer> tokenFreqs) {
        ArrayList<String> symbols = new ArrayList<>();
        symbols.add(maxFreqPair.getKey() + maxFreqPair.getValue());

        HashMap<String, Integer> newTokenFreqs = new HashMap<>();
        for (Map.Entry<String, Integer> e : tokenFreqs.entrySet()) {
            String token = e.getKey();
            String newToken =
                    token.replace(
                            maxFreqPair.getKey() + " " + maxFreqPair.getValue(),
                            maxFreqPair.getKey() + "" + maxFreqPair.getValue());
            newTokenFreqs.put(newToken, tokenFreqs.get(token));
        }
        return new Pair(newTokenFreqs, symbols.toArray(new String[symbols.size()]));
    }

    public static List<String> segmentBPE(String[] tokens, String[] symbols) {
        List<String> outputs = new ArrayList<>();
        for (String token : tokens) {
            int start = 0;
            int end = token.length();
            ArrayList<String> curOutput = new ArrayList<>();
            // Segment token with the longest possible subwords from symbols
            while (start < token.length() && start < end) {
                if (Arrays.asList(symbols).contains(token.substring(start, end))) {
                    curOutput.add(token.substring(start, end));
                    start = end;
                    end = token.length();
                } else {
                    end -= 1;
                }
            }
            if (start < tokens.length) {
                curOutput.add("[UNK]");
            }
            String temp = "";
            for (String s : curOutput) {
                temp += s + " ";
            }
            outputs.add(temp.trim());
        }
        return outputs;
    }
}
