from EmotionGestureCompiler import EmotionGestureCompiler

def main ():
    emo = EmotionGestureCompiler(
        model_name = "resnet18.onnx",
        model_option = "onnx",
        backend_option = 1,
        providers = 1,
        fp16 = False,
        num_faces = 1,
        train_path = 'Base_de_dados',
        k = 7,
        video = True
    )
    emo.video(0)

if __name__ == "__main__":
    main()