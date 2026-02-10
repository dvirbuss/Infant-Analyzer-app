from dataclasses import dataclass
from typing import List, Dict

@dataclass
class AimsScores:
    prone: List[bool]
    supine: List[bool]
    sitting: List[bool]
    standing: List[bool]

    def total(self) -> int:
        return sum(map(int, self.prone + self.supine + self.sitting + self.standing))

def init_scores() -> AimsScores:
    prone   = [True, True] + [False]*9
    supine  = [True, True] + [False]*6
    sitting = [True] + [False]*6
    standing= [True] + [False]*2
    return AimsScores(prone, supine, sitting, standing)

def score_all(keypoints_tsv_path: str, angles_tsv_path: str) -> AimsScores:
    scores = init_scores()

    # call scoring rules (pure functions) that update `scores`
    # prone_total(scores, keypoints_tsv_path, angles_tsv_path)
    # supine_total(scores, ...)
    # sitting_total(scores, ...)
    # standing_total(scores, ...)

    return scores
