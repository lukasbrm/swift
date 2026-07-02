package cc.briem.swift;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.image.ImageView;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

public class DisplayApp extends Application {

    static volatile ImageView imageView;

    @Override
    public void start(Stage primaryStage) {
        imageView = new ImageView();
        imageView.setPreserveRatio(true);
        imageView.setSmooth(true);

        StackPane root = new StackPane(imageView);
        Scene scene = new Scene(root, 1280, 720);

        imageView.fitWidthProperty().bind(scene.widthProperty());
        imageView.fitHeightProperty().bind(scene.heightProperty());

        primaryStage.setTitle("Swift Frame Viewer");
        primaryStage.setScene(scene);
        primaryStage.show();
    }
}
