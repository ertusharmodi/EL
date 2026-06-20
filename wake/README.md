# Drop a trained .onnx wake word model here and update WAKE_MODEL in config.py.
#
# To train a custom "Hey Eleven" model:
#   1. Open the OpenWakeWord training notebook:
#      https://github.com/dscripka/openWakeWord#training-new-models
#   2. Enter "hey eleven" as the target phrase.
#   3. Run all cells (~30 minutes on a free Colab GPU).
#   4. Download the generated hey_eleven.onnx file and place it in this directory.
#   5. In config.py, change:
#        WAKE_MODEL = "hey_jarvis"
#      to:
#        WAKE_MODEL = os.path.join(_HERE, "wake", "hey_eleven.onnx")
#
# No other code changes are needed.
