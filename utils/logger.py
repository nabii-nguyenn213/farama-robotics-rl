import os 
import csv 
import time 
import copy 
import yaml 
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

class Logger: 

    def __init__(self, config, run_name, log_dir=None, tb_dir=None): 
        self.config = config 
        self.run_name = run_name
        env_name = config["env"].get("name", "FetchSlide-v4")
        self.log_dir = config["dir"].get("log", f"./logs/log/{env_name}/{run_name}") if log_dir is None else log_dir
        self.tb_dir = config["dir"].get("tensorboard", f"./logs/tensorboard_logs/{env_name}/{run_name}") if tb_dir is None else tb_dir

        self.start_time = time.time()
        self.best_eval_return = float("-inf")

        self.text_log_path = os.path.join(self.log_dir, f"{self.run_name}.log")
        self.train_csv_path = os.path.join(self.log_dir, f"{self.run_name}_train.csv")
        self.episode_csv_path = os.path.join(self.log_dir, f"{self.run_name}_episode.csv")
        self.eval_csv_path = os.path.join(self.log_dir, f"{self.run_name}_eval.csv")
        self.config_save_path = os.path.join(self.log_dir, f"{self.run_name}_config.yaml")
        
        self.writer = SummaryWriter(log_dir=self.tb_dir)

    def _init_files(self):
        if not os.path.exists(self.train_csv_path):
            with open(self.train_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step",
                    "critic_loss",
                    "q1_loss",
                    "q2_loss",
                    "actor_loss",
                    "q1_mean",
                    "q2_mean",
                    "log_pi_mean",
                ])

        if not os.path.exists(self.episode_csv_path):
            with open(self.episode_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "step",
                    "episodic_return",
                    "episode_length",
                    "elapsed_sec",
                ])

        if not os.path.exists(self.eval_csv_path):
            with open(self.eval_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step",
                    "avg_return",
                    "best_so_far",
                    "elapsed_sec",
                ])

        if not os.path.exists(self.text_log_path):
            with open(self.text_log_path, "w", encoding="utf-8") as f:
                f.write("")

    def _to_float(self, x):
        try:
            return float(x)
        except Exception:
            return x

    def _append_text(self, msg):
        with open(self.text_log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n") 

    def _elapsed_sec(self):
        return time.time() - self.start_time

    def save_config(self):
        config_to_save = self.config

        try:
            config_to_save = OmegaConf.to_container(self.config, resolve=True)
        except Exception:
            pass

        with open(self.config_save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_to_save, f, sort_keys=False)

    def log_train(self, step, metrics: dict, print_to_console=False):
        critic_loss = self._to_float(metrics.get("critic_loss", 0.0))
        q1_loss = self._to_float(metrics.get("q1_loss", 0.0))
        q2_loss = self._to_float(metrics.get("q2_loss", 0.0))
        actor_loss = self._to_float(metrics.get("actor_loss", 0.0))
        q1_mean = self._to_float(metrics.get("q1_mean", 0.0))
        q2_mean = self._to_float(metrics.get("q2_mean", 0.0))
        log_pi_mean = self._to_float(metrics.get("log_pi_mean", 0.0))

        with open(self.train_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                critic_loss,
                q1_loss,
                q2_loss,
                actor_loss,
                q1_mean,
                q2_mean,
                log_pi_mean,
            ])

        if self.writer is not None:
            self.writer.add_scalar("train/critic_loss", critic_loss, step)
            self.writer.add_scalar("train/q1_loss", q1_loss, step)
            self.writer.add_scalar("train/q2_loss", q2_loss, step)
            self.writer.add_scalar("train/actor_loss", actor_loss, step)
            self.writer.add_scalar("train/q1_mean", q1_mean, step)
            self.writer.add_scalar("train/q2_mean", q2_mean, step)
            self.writer.add_scalar("train/log_pi_mean", log_pi_mean, step)

        if print_to_console:
            self.info(
                f"step={step} "
                f"critic_loss={critic_loss:.4f} "
                f"actor_loss={actor_loss:.4f} "
                f"log_pi={log_pi_mean:.4f} "
                f"q1={q1_mean:.4f} "
                f"q2={q2_mean:.4f}"
            )

    def info(self, msg): 
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg)
        self._append_text(full_msg)

    def log_episode(self, episode, step, episodic_return, episode_length):
        elapsed = self._elapsed_sec()

        with open(self.episode_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                step,
                self._to_float(episodic_return),
                self._to_float(episode_length),
                elapsed,
            ])

        if self.writer is not None:
            self.writer.add_scalar("episode/return", self._to_float(episodic_return), step)
            self.writer.add_scalar("episode/length", self._to_float(episode_length), step)

        self.info(
            f"episode={episode} "
            f"step={step} "
            f"return={float(episodic_return):.3f} "
            f"length={int(episode_length)}"
        )

    def log_eval(self, step, avg_return):
        avg_return = self._to_float(avg_return)
        is_best = avg_return > self.best_eval_return
        if is_best:
            self.best_eval_return = avg_return

        elapsed = self._elapsed_sec()

        with open(self.eval_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                avg_return,
                self.best_eval_return,
                elapsed,
            ])

        if self.writer is not None:
            self.writer.add_scalar("eval/avg_return", avg_return, step)
            self.writer.add_scalar("eval/best_return", self.best_eval_return, step)

        if is_best:
            self.info(f"[eval] step={step} avg_return={avg_return:.3f} NEW_BEST")
        else:
            self.info(f"[eval] step={step} avg_return={avg_return:.3f}")

        return is_best

    def log_checkpoint(self, path, step=None, kind="checkpoint"):
        if step is None:
            self.info(f"saved {kind}: {path}")
        else:
            self.info(f"saved {kind} at step={step}: {path}")

    def close(self):
        self.info("Closing logger")
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
