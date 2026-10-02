"""Base class for scenario packs.

A pack is a thin layer of domain knowledge on top of the generic engine:
  * canonical stage names -> roles (a stage template),
  * config validation and derived values (e.g. a hop period),
  * default requirements (deadline/period) derived from that config,
  * extra metrics and diagnosis rules in the scenario's own language.
Packs never measure anything themselves; they interpret the generic result.
"""


class Pack:
    name = "custom"
    description = "Any pipeline: arbitrary named stages, generic metrics and rules."
    stage_roles = {}          # canonical stage name -> role

    def configure(self, config):
        return dict(config)

    def role_for(self, stage_name):
        """Role for a stage name, or None to let the engine infer it."""
        return self.stage_roles.get(stage_name)

    def default_deadline_ms(self, config):
        return None

    def default_period_ms(self, config):
        return None

    def metrics(self, result):
        return {}

    def findings(self, result):
        return []

    def template(self):
        return list(self.stage_roles.items())


class CustomPack(Pack):
    pass
