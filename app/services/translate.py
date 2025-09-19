from deep_translator import GoogleTranslator

def translate_text(txt, src, dest):
    return GoogleTranslator(source=src, target=dest).translate(txt)