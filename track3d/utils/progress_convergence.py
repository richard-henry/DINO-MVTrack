"""A3.3 follow-up: positive learning-rate floor and loss-only checkpoint choice.

Checkpoint choice uses post-update export-mode objective and input coverage.
Learning-rate decisions use the actual pre-update training objective windows.
"""
import math
from track3d.utils.loss_convergence import ProtocolSchedule, LossPlateau


class ProgressSchedule(ProtocolSchedule):
    def __init__(self, optimizer, schedule, total_steps, scales):
        self.current_peak = schedule['peak_lr']
        super().__init__(optimizer, schedule, total_steps, scales)

    def base_lr(self, update):
        s=self.schedule
        if update<=s['warmup']:
            return s['floor_lr']+(s['peak_lr']-s['floor_lr'])*update/max(1,s['warmup'])
        return self.current_peak

    def reduce(self):
        self.current_peak=max(self.schedule['floor_lr'],self.current_peak*self.schedule['drop_factor'])
        self._apply()

    def restart_at(self, step):
        self.current_peak=self.schedule['peak_lr']
        super().restart_at(step)


class ProgressMonitor(LossPlateau):
    def __init__(self, recipe):
        super().__init__(recipe['stopping'])
        self.lr_config=dict(recipe['schedule'])
        self.progress_reference=None
        self.bad_windows=0
        self.floor_streak=0
        self.requested_reduction=False
        self.selected_step=None
        self.selected_loss=None
        self.initial_coverage=None
        self.checkpoint_scores=[]
        self.reductions=[]

    def consider_checkpoint(self, step, diagnostics):
        loss=diagnostics['total']; coverage=diagnostics['coverage']
        old=next((r for r in self.checkpoint_scores if r['step']==step),None)
        if old is not None:
            if old['loss']!=loss:raise ValueError('Restored checkpoint objective differs')
            return
        if self.initial_coverage is None:self.initial_coverage=list(coverage)
        drop=max(a-b for a,b in zip(self.initial_coverage,coverage))
        eligible=math.isfinite(loss) and all(c>0 for c in coverage) and drop<=self.config['coverage_drop']
        self.checkpoint_scores.append(dict(step=step,loss=loss,coverage_drop=drop,eligible=eligible))
        if eligible and (self.selected_loss is None or loss<self.selected_loss):
            self.selected_step=step;self.selected_loss=loss

    def observe(self, step, loss, diagnostics, lr):
        self.requested_reduction=False
        count=len(self.checks)
        super().observe(step,loss,diagnostics)
        if len(self.checks)==count:return False
        check=self.checks[-1]; mean=check['last_mean']; s=self.lr_config
        support=check['checks']['coverage_drop'] and check['checks']['coverage_change'] and all(c>0 for c in diagnostics['coverage'])
        improvement=self.progress_reference is None or self.progress_reference-mean>max(abs(self.progress_reference),1e-8)*self.config['relative_change']
        if support and improvement:
            self.progress_reference=mean;self.bad_windows=0
        else:self.bad_windows+=1
        at_floor=lr<=s['floor_lr']*(1+1e-9)
        self.floor_streak=self.floor_streak+1 if at_floor and support and all(check['checks'].values()) else 0
        if len(self.rows)>=s['min_lr_updates'] and self.bad_windows>=s['lr_patience_windows'] and not at_floor:
            self.requested_reduction=True;self.bad_windows=0;self.floor_streak=0
            self.reductions.append(dict(after_step=step,old_lr=lr,new_lr=max(s['floor_lr'],lr*s['drop_factor'])))
        check.update(progress_reference=self.progress_reference,bad_windows=self.bad_windows,
                     learning_rate=lr,at_floor=at_floor,floor_streak=self.floor_streak,
                     requested_reduction=self.requested_reduction)
        return (len(self.rows)>=self.config['min_updates'] and at_floor and
                self.bad_windows>=s['lr_patience_windows'] and self.floor_streak>=self.config['patience'])

    def state_dict(self):
        return dict(super().state_dict(),progress={k:v for k,v in self.__dict__.items()
                    if k not in ('config','rows','checks','streak')})

    def load_state_dict(self, state):
        if state['progress']['lr_config']!=self.lr_config:raise ValueError('Progress LR configuration differs')
        super().load_state_dict(state);self.__dict__.update(state['progress'])

    def report(self, reason, step):
        return dict(super().report(reason,step),selected_step=self.selected_step,selected_loss=self.selected_loss,
                    checkpoint_scores=self.checkpoint_scores,reductions=self.reductions,
                    selection_contract='post_update_eval_objective_with_input_coverage_guard')
