from pathlib import Path
import importlib
import pkgutil


class SkillManager:
    def __init__(self):
        self.skills = []

    def load(self):
        self.skills.clear()

        package = "learned_skills"

        for _, module_name, _ in pkgutil.walk_packages(
                Path("learned_skills").resolve().parent.iterdir(),
                package + "."):

            try:
                module = importlib.import_module(module_name)

                if hasattr(module, "SKILL"):
                    self.skills.append(module.SKILL)

            except Exception as e:
                print(f"Failed loading {module_name}: {e}")

    def applicable_skills(self, task):
        applicable = []

        for skill in self.skills:
            if skill.applicable(task):
                applicable.append(skill)

        return applicable

    def learn(self, train_pairs):
        learned = []

        for skill in self.skills:

            result = skill.learn(train_pairs)

            if result.success:
                learned.append(result)

        return learned