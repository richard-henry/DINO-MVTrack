"""Input-only TTO loss accounting, explicit LR schedules and plateau detection.

No GT metrics or trajectory error enters this module. A plateau is an operational
stopping condition, not a statement about global optimality or tracking accuracy.
"""
import json
import math
import statistics
from pathlib import Path


def load_recipe(path):
    if not path:
        return None
    recipe = json.loads(Path(path).read_text())
    schedule = recipe['schedule']
    if schedule['kind'] not in ('constant', 'warm_cosine', 'legacy', 'loss_plateau'):
        raise ValueError('Unknown convergence schedule')
    if schedule['kind'] != 'legacy':
        if not all(math.isfinite(schedule[k]) for k in ('floor_lr','peak_lr')) or not 0 < schedule['floor_lr'] <= schedule['peak_lr']:
            raise ValueError('Require positive floor <= peak learning rate')
        if schedule['kind'] == 'warm_cosine' and not 0 <= schedule['warmup'] < schedule['decay_steps']:
            raise ValueError('Invalid warmup/decay interval')
        if schedule['kind'] == 'loss_plateau':
            if not 0 < schedule['drop_factor'] < 1 or schedule['lr_patience_windows'] < 1 or schedule['warmup'] < 0:
                raise ValueError('Invalid progress-driven schedule')
    if recipe.get('restart_schedule') and schedule['kind']=='legacy':
        raise ValueError('Legacy scheduler cannot be restarted through the convergence recipe')
    s = recipe['stopping']
    for key in ('window','patience','min_updates'):
        if type(s[key]) is not int:
            raise ValueError('Stopping counts must be integers: '+key)
    for key in ('warmup','decay_steps','lr_patience_windows','min_lr_updates'):
        if key in schedule and type(schedule[key]) is not int:
            raise ValueError('Schedule counts must be integers: '+key)
    if s['window'] < 2 or s['patience'] < 1 or s['min_updates'] < 2*s['window']:
        raise ValueError('Invalid loss windows')
    for key in ('relative_change', 'cv', 'component_change', 'coverage_change', 'coverage_drop'):
        if not math.isfinite(s[key]) or s[key] < 0:
            raise ValueError('Invalid threshold: '+key)
    return recipe


class ProtocolSchedule:
    """LR for the NEXT update; constant positive floor survives the decay horizon."""
    def __init__(self, optimizer, schedule, total_steps, scales):
        self.optimizer = optimizer
        self.schedule = dict(schedule)
        self.total_steps = total_steps
        self.scales = list(scales)
        self.origin = 0
        self.last_epoch = 0
        self._apply()

    def base_lr(self, update):
        s = self.schedule
        if s['kind'] == 'constant':
            return s['peak_lr']
        if update <= s['warmup']:
            return s['floor_lr'] + (s['peak_lr']-s['floor_lr'])*update/max(s['warmup'], 1)
        progress = min(1., max(0., (update-s['warmup'])/(s['decay_steps']-s['warmup'])))
        return s['floor_lr'] + .5*(s['peak_lr']-s['floor_lr'])*(1+math.cos(math.pi*progress))

    def _apply(self):
        self._last_lr = [self.base_lr(self.last_epoch-self.origin+1)*scale for scale in self.scales]
        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group['lr'] = lr

    def restart_at(self, step):
        self.origin = self.last_epoch = step
        self._apply()

    def step(self):
        self.last_epoch += 1
        self._apply()

    def state_dict(self):
        return {k:v for k,v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state):
        self.__dict__.update(state)
        self._apply()


def summarize_iterations(rows, anchor, anchor_weight, total):
    # rows contain detached Python numbers, so retaining diagnostics holds no graph.
    result = dict(schema=2, iterations=rows, total=float(total), anchor=float(anchor),
                  weighted_anchor=float(anchor)*anchor_weight)
    for key in ('weighted_rep', 'weighted_smooth', 'weighted_rigid'):
        result[key] = sum(r[key] for r in rows)
    result['reconstructed_total'] = sum(result[k] for k in
        ('weighted_rep', 'weighted_smooth', 'weighted_rigid', 'weighted_anchor'))
    result['reconstruction_error'] = result['reconstructed_total']-result['total']
    result['coverage'] = [r['valid_weight']/max(r['semantic_weight'], 1e-12) for r in rows]
    return result


class LossPlateau:
    def __init__(self, config):
        self.config = dict(config)
        self.rows = []
        self.checks = []
        self.streak = 0

    def state_dict(self):
        return dict(config=self.config, rows=self.rows, checks=self.checks, streak=self.streak)

    def load_state_dict(self, state):
        if state['config'] != self.config:
            raise ValueError('Loss stopping configuration differs')
        self.rows, self.checks, self.streak = state['rows'], state['checks'], state['streak']

    def observe(self, step, loss, diagnostics):
        if not math.isfinite(loss):
            raise ValueError('Non-finite objective')
        self.rows.append(dict(step=step, loss=loss, diagnostics=diagnostics))
        c = self.config; w = c['window']; n = len(self.rows)
        if n < 2*w or n % w:
            return False
        a, b = self.rows[-2*w:-w], self.rows[-w:]
        mean = lambda rows, f: statistics.mean(f(r) for r in rows)
        am, bm = mean(a, lambda r:r['loss']), mean(b, lambda r:r['loss'])
        scale = max(abs(am), 1e-8)
        change = (bm-am)/scale
        cv = statistics.pstdev(r['loss'] for r in b)/max(abs(bm), 1e-8)
        keys = ('weighted_rep', 'weighted_smooth', 'weighted_rigid', 'weighted_anchor')
        if 'weighted_auxiliary' in diagnostics:
            keys += ('weighted_auxiliary',)
        changes = {k:(mean(b,lambda r:r['diagnostics'][k])-mean(a,lambda r:r['diagnostics'][k]))/scale for k in keys}
        initial = self.rows[:w]
        coverage_change = max(abs(mean(b,lambda r:r['diagnostics']['coverage'][i])-mean(a,lambda r:r['diagnostics']['coverage'][i])) for i in range(len(diagnostics['coverage'])))
        coverage_drop = max(mean(initial,lambda r:r['diagnostics']['coverage'][i])-mean(b,lambda r:r['diagnostics']['coverage'][i]) for i in range(len(diagnostics['coverage'])))
        checks = dict(loss=abs(change)<=c['relative_change'], cv=cv<=c['cv'],
                      components=max(map(abs,changes.values()))<=c['component_change'],
                      coverage_change=coverage_change<=c['coverage_change'], coverage_drop=coverage_drop<=c['coverage_drop'],
                      supervision=all(value>0 for value in diagnostics['coverage']))
        passed = all(checks.values()) and n>=c['min_updates']
        self.streak = self.streak+1 if passed else 0
        self.checks.append(dict(step=step, updates=n, previous_mean=am, last_mean=bm, relative_change=change,
                               cv=cv, component_changes=changes, coverage_change=coverage_change,
                               coverage_drop=coverage_drop, checks=checks, streak=self.streak))
        return self.streak>=c['patience']

    def report(self, reason, step):
        return dict(schema=1, reason=reason, final_step=step, observed_updates=len(self.rows),
                    loss_only=True, configuration=self.config, checks=self.checks)
