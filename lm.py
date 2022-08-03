import torch
from transformers import RobertaTokenizer, RobertaModel


PLM_MODEL = "microsoft/codebert-base"


def nlpl2vec(nlpl_list: list[str],
             tokenizer: RobertaTokenizer,
             model: RobertaModel):
    """

    :param nlpl_list: .
    :param tokenizer: Codebert RobertaTokenizer object from HF.
    :param model: Codebert RobertaModel object from HF.
    :return:
    """
    with torch.no_grad():

        tokens = [tokenizer.cls_token]

        for s in nlpl_list:
            tokens.extend(tokenizer.tokenize(s))
            tokens.append(tokenizer.sep_token)

        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
        context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]

    return context_embeddings

