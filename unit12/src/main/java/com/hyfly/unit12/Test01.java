package com.hyfly.unit12;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

@Slf4j
public class Test01 {

    public static void main(String[] args) throws Exception {
        // Load the original image
        Image imgArr = ImageFactory.getInstance()
                .fromUrl("https://github.com/d2l-ai/d2l-en/blob/master/img/catdog.jpg?raw=true");
        imgArr.getWrappedImage();

        // bbox is the abbreviation for bounding box
        double[] dog_bbox = new double[]{60, 45, 378, 516};
        double[] cat_bbox = new double[]{400, 112, 655, 493};

        List<String> classNames = new ArrayList();
        classNames.add("dog");
        classNames.add("cat");

        List<Double> prob = new ArrayList<>();
        prob.add(1.0);
        prob.add(1.0);

        List<BoundingBox> boxes = new ArrayList<>();
        boxes.add(bboxToRectangle(dog_bbox, imgArr.getWidth(), imgArr.getHeight()));
        boxes.add(bboxToRectangle(cat_bbox, imgArr.getWidth(), imgArr.getHeight()));

        DetectedObjects detectedObjects = new DetectedObjects(classNames, prob, boxes);

        // drawing the bounding boxes on the original image
        imgArr.drawBoundingBoxes(detectedObjects);
        imgArr.getWrappedImage();
    }

    public static Rectangle bboxToRectangle(double[] bbox, int width, int height) {
        // Convert the coordinates into the
        // bounding box coordinates format
        return new Rectangle(bbox[0] / width, bbox[1] / height, (bbox[2] - bbox[0]) / width, (bbox[3] - bbox[1]) / height);
    }
}
