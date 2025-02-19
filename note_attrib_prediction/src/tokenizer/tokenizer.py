from pytorch_lightning import LightningModule
from typing import List
from miditok.vocabulary import Vocabulary
from dataclasses import dataclass
import hydra
from omegaconf import DictConfig
from itertools import chain


@dataclass
class Event:
    type: str
    value: int
    vocab = 1


    def __str__(self) -> str:
        return f"{self.type}_{self.value}"
    
    def __repr__(self) -> str:
        return f"Event(type={self.type}, value={self.value})"
    

@dataclass
class String_Fret_Event:
    string: int
    fret: int
    pitch: int
    vocab = 3

    def __str__(self) -> str:
        return f"String_{self.string}_Fret_{self.fret}_Pitch_{self.pitch}"
    
    def __repr__(self) -> str:
        return f"Event(string={self.string}, fret={self.fret}, note={self.pitch})"


class tokenizer_initializer(LightningModule):
    def __init__(
        self,
        pitch_range:dict,
        pos_range:dict,
        dur_range:dict,
        string_range:dict,
        fret_range:dict,
        mask: bool,
        sos_eos: bool
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        pitch_range = range(pitch_range["start"], pitch_range["end"] + 1)
        pos_range = range(pos_range["start"], pos_range["end"] + 1)
        #dur_range = [0, 1]
        dur_range = range(dur_range["start"], dur_range["end"] + 1)
        string_range = range(string_range["start"], string_range["end"] + 1)
        fret_range = range(fret_range["start"], fret_range["end"] + 1)
        self.tokenizer = TabTokenizer(
            pitch_range,
            pos_range,
            dur_range,
            string_range,
            fret_range,
            mask=mask,
            sos_eos=sos_eos,
        )


    def generate(self):
        self.vocab = self.tokenizer._create_vocabulary()
        self.vocab_size = len(self.vocab._event_to_token)
        return self.tokenizer, self.vocab, self.vocab_size


class TabTokenizer:
    def __init__(
        self,
        pitch_range,
        pos_range,
        dur_range,
        string_range,
        fret_range,
        pad: bool=True,
        sos_eos: bool=False,
        mask: bool=False
    ):
        self.pitch_range = pitch_range
        self.pos_range = pos_range
        self.dur_range = dur_range
        self.string_range = string_range
        self.fret_range = fret_range
        self.pad = pad
        self.sos_eos = sos_eos
        self.mask = mask


    def tokens_to_events(self, tokens: List[int]) -> List[Event]:
        events = []
        vocab = self._create_vocabulary()
        tokens = list(chain.from_iterable(tokens))
        for token in tokens:
            event = vocab.token_to_event[token].split("_")
            if len(event) == 6:
                events.append(String_Fret_Event(event[1], event[3], event[5]))
            elif len(event) == 2:
                events.append(Event(event[0], event[1]))
        return events


    def tab_to_tokens(self, tab) -> List[int]:
        events = []
        vocab = self._create_vocabulary()
        for pos, dur, pitch, string, fret in tab:
            events.append(Event(type="Pos", value=pos))
            events.append(Event(type="Dur", value=dur))
            events.append(String_Fret_Event(string=string, fret=fret, pitch=pitch))
        return [vocab.event_to_token[str(event)] for event in events]
    

    def tokens_to_tab(self, tokens: List[int]) -> List[List[int]]:
        tab = []
        events = self.tokens_to_events(tokens)
        for i, event in enumerate(events):
            if i >= len(events) - 2:
                break
            elif event.vocab == 1 and events[i + 1].vocab == 1 and events[i + 2].vocab == 3:
                if event.type == "Pos" and events[i + 1].type == "Dur":
                    tab.append([event.value, events[i + 1].value, events[i + 2].pitch, events[i + 2].string, events[i + 2].fret])
        return tab
    
    
    def _create_vocabulary(self) -> Vocabulary:
        vocab = Vocabulary(pad=self.pad, sos_eos=self.sos_eos, mask=self.mask)
        open_string_pitch = [76, 71, 67, 62, 57, 52]
        for i in self.string_range[1:]:
            for j in self.fret_range[1:]:
                pitch = open_string_pitch[i - 1] + j
                vocab.add_event(f"String_{i}_Fret_{j}_Pitch_{pitch}")
            vocab.add_event(f"String_{i}_Fret_-1_Pitch_101")
        vocab.add_event("String_0_Fret_-1_Pitch_100")
        vocab.add_event(f"Pos_{i}" for i in self.pos_range)
        vocab.add_event(f"Dur_{i}" for i in self.dur_range)
        return vocab


@hydra.main(version_base="1.3", config_path='../../conf/tokenizer', config_name='tokenizer.yaml')
def main(cfg: DictConfig) -> None:
    initializer = tokenizer_initializer(
        pitch_range = cfg.pitch_range,
        pos_range = cfg.pos_range,
        dur_range = cfg.dur_range,
        string_range = cfg.string_range,
        fret_range=cfg.fret_range,
        mask = cfg.mask,
        sos_eos = cfg.sos_eos,
    )
    tokenizer, vocab, vocab_size = initializer.generate()
    print(vocab_size)
    for i in range(vocab_size):
        print(i, vocab[i])


if __name__ == "__main__":
    main()
