from EmotionGestureCompiler import EmotionGestureCompiler

def main ():
    emo = EmotionGestureCompiler(
        gestures = ['A', 'B', 'C', 'D', 'E'],
        model_name = "resnet18.onnx",
        model_option = "onnx",
        backend_option = 1,
        providers = 1,
        fp16 = False,
        num_faces = 1,
        train_path = 'Base_de_dados',
        database_path = 'Base_de_dados',
        k = 7
    )
    emo.video(0)

if __name__ == "__main__":
    main()