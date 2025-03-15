from dataclasses import dataclass

@dataclass
class ClassItem:
    index: int
    name: str

@dataclass
class Classes:
    def __init__(self, class_items: list[ClassItem]):
        self.class_items = class_items
        for item in class_items:
            setattr(self, item.name, item)