# This is a sample Python script.
import LanguageFineTuning
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import LanguageFineTuning.MaskedLanguageModeling as MLM
from datasets import load_dataset


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
    model_checkpoint = "distilroberta-base"
    model = MLM.MaskedLanguageModeling(datasets, model_checkpoint)
    model.show_train_data()
    model.decode_text_for_language_model()
    model.train()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
