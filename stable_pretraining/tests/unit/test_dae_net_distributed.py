"""Real CPU/Gloo checks for the DAE-Net global objective and exposure shards."""

from datetime import timedelta
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from stable_pretraining.data.dae_net import (
    DAENetDistributedBatchSampler,
    DAENetExposureSampler,
)
from stable_pretraining.losses.dae_net import DAENetLoss

pytestmark = pytest.mark.unit


class _Outputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 6)

    def forward(self, inputs):
        value = self.linear(inputs).sigmoid()
        raw = value[:, :3, None].transpose(1, 2).expand(-1, 5, -1)
        return {
            "occupancy_sum": raw.sum(-1, keepdim=True),
            "occupancy_max": raw.max(-1, keepdim=True).values,
            "raw_occupancy": raw,
            "existence": value[:, 3:, None],
            "affine": value[:, :3, None].expand(-1, -1, 12),
            "deformation": value[:, :3, None, None].expand(-1, -1, 3, 5),
        }


def _launch(worker, directory, *args):
    context = mp.spawn(
        worker, args=(str(directory / "rendezvous"), *args), nprocs=2, join=False
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("DAE-Net two-process test exceeded 120 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()


def _global_reference(output, target, sparse):
    # Evaluate the global equations independently of DAENetLoss's reductions.
    inside = target > 0.5
    reconstruction = []
    for key, weight in (("occupancy_sum", 1.0), ("occupancy_max", 0.1)):
        value = output[key]
        total = (1 - value[inside]).square().sum()
        total = total + value[~inside].clamp_min(0.1).square().sum()
        reconstruction.append(weight * total / target.numel())
    existence = output["existence"]
    sparsity = (
        -sum(
            (1 - existence[:, branch].sum() / target.shape[0]).square()
            for branch in range(existence.shape[1])
        )
        / existence.shape[1]
    )
    masked = (
        output["deformation"]
        * output["raw_occupancy"].detach().permute(0, 2, 1)[:, :, None]
    )
    terms = dict(
        reconstruction_sum=reconstruction[0],
        reconstruction_max=reconstruction[1],
        sparsity=sparsity * sparse * 0.37,
        affine=output["affine"].square().sum() / output["affine"].numel() * 0.13 * 0.37,
        deformation=masked.square().sum() / masked.numel() * 0.41 * 0.37,
    )
    return {"loss": sum(terms.values()), **terms}


def _loss_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        for sizes in ((2, 2), (3, 2)):
            for sparse in (0.0, 0.7):
                torch.manual_seed(51)
                reference = _Outputs()
                model = DDP(_Outputs())
                model.module.load_state_dict(reference.state_dict())
                objective = DAENetLoss(
                    sparse_weight=sparse, affine_weight=0.13, deform_weight=0.41
                )
                inputs = torch.linspace(-3, 4, sum(sizes) * 3).reshape(-1, 3)
                target = torch.randint(0, 2, (sum(sizes), 5, 1)).float()
                start = sum(sizes[:rank])
                shard = slice(start, start + sizes[rank])
                expected = _global_reference(reference(inputs), target, sparse)
                actual = objective(
                    model(inputs[shard]), target[shard], 0.37, distributed=True
                )
                assert actual["global_batch_size"].item() == sum(sizes)
                for name, value in expected.items():
                    key = "loss_report" if name == "loss" else name
                    torch.testing.assert_close(actual[key], value, rtol=1e-5, atol=1e-6)
                if sparse:
                    local = objective(model.module(inputs[shard]), target[shard], 0.37)
                    wrong = local["sparsity"].detach().clone()
                    dist.all_reduce(wrong)
                    assert not torch.isclose(wrong / 2, expected["sparsity"], atol=1e-6)
                expected["loss"].backward()
                actual["loss"].backward()
                for parameter, ref in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(
                        parameter.grad, ref.grad, rtol=1e-5, atol=1e-6
                    )
                torch.optim.Adam(reference.parameters(), lr=2e-4).step()
                torch.optim.Adam(model.parameters(), lr=2e-4).step()
                for parameter, ref in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter, ref, rtol=1e-5, atol=1e-6)
                    other = parameter.detach().clone()
                    dist.broadcast(other, src=0)
                    assert torch.equal(parameter, other)
                # Offline scoring inside a live group must never communicate.
                if rank == 0:
                    objective(reference(inputs), target)
                dist.barrier()
    finally:
        dist.destroy_process_group()


def test_global_loss_ddp_parameter_gradients_and_adam(tmp_path):
    _launch(_loss_worker, tmp_path)


class _Records:
    def __init__(self, count):
        self.records = [{"query_resolution": 64} for _ in range(count)]

    def __len__(self):
        return len(self.records)


@pytest.mark.parametrize(
    "count,quota,batch_size", [(129, 512, 16), (5, 11, 6), (5, 5, 3)]
)
def test_rank_shards_reconstruct_global_batches(count, quota, batch_size):
    sampler = DAENetExposureSampler(_Records(count), shapes_per_epoch=quota, seed=19)
    ranks = [
        DAENetDistributedBatchSampler(sampler, batch_size, rank=r, world_size=2)
        for r in range(2)
    ]
    for epoch in (0, 1, 7):
        for rank in ranks:
            rank.set_epoch(epoch)
        expected = list(sampler)
        actual = []
        for step, shards in enumerate(zip(*ranks)):
            batch = shards[0] + shards[1]
            assert batch == expected[step * batch_size : (step + 1) * batch_size]
            assert min(map(len, shards)) > 0
            actual.extend(batch)
        assert actual == expected
        assert len(list(ranks[0])) == len(ranks[0])


@pytest.mark.parametrize("count,batch_size", [(5, 4), (3, 1)])
def test_empty_rank_batch_rejected_before_iteration(count, batch_size):
    sampler = DAENetExposureSampler(_Records(count), shapes_per_epoch=count)
    for rank in range(2):
        batches = DAENetDistributedBatchSampler(
            sampler, batch_size, rank=rank, world_size=2
        )
        with pytest.raises(ValueError, match="at least world_size"):
            next(iter(batches))


def test_lightning_updates_nested_exposure_sampler():
    from lightning.fabric.utilities.data import _set_sampler_epoch
    from torch.utils.data import DataLoader

    records = _Records(5)
    sampler = DAENetExposureSampler(records, shapes_per_epoch=5)
    batches = DAENetDistributedBatchSampler(sampler, 3, rank=0, world_size=2)
    loader = DataLoader(records, batch_sampler=batches)
    _set_sampler_epoch(loader, 7)
    assert sampler.epoch == 7


def _controller_worker(rank, rendezvous):
    from stable_pretraining.backbone.dae_net import DAENetBackbone
    from stable_pretraining.callbacks.dae_net import DAENetRevival

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        torch.manual_seed(71)
        backbone = DAENetBackbone(branch_num=3, z_dim=4, ef_dim=2, df_dim=6, gf_dim=8)
        controller = DAENetRevival(era_length=2)
        module = SimpleNamespace(
            backbone=backbone, device=torch.device("cpu"), log=lambda *a, **k: None
        )
        data = SimpleNamespace(
            usage_dataloader=lambda: [{"voxels": torch.zeros(2, 1, 64, 64, 64)}]
        )
        trainer = SimpleNamespace(
            world_size=2,
            is_global_zero=rank == 0,
            datamodule=data,
            global_rank=rank,
            current_epoch=1,
        )
        prediction = torch.tensor([0.0] if rank == 0 else [1.0, 1.0])
        target = torch.tensor([0.0] if rank == 0 else [1.0, 0.0])
        assert (
            controller.batch_reconstruction_iou(prediction, target, distributed=True)
            == 0.5
        )
        for pred, truth in (
            (torch.zeros(1), torch.zeros(1)),
            (torch.tensor([float("nan") if rank else 0.0]), torch.ones(1)),
        ):
            with pytest.raises(ValueError):
                controller.batch_reconstruction_iou(pred, truth, distributed=True)
        # Forced scores cover the decision branches independently of natural training.
        controller._era_iou_sum, controller._era_iou_batches = 0.8, 1
        controller.on_train_epoch_end(trainer, module)
        accepted = {k: v.clone() for k, v in backbone.state_dict().items()}
        optimizer = torch.optim.Adam(backbone.parameters())
        for parameter in backbone.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        assert optimizer.state
        parameter_ids = [id(p) for p in backbone.parameters()]
        for usage, ages in (
            (torch.tensor([0.0, 2.0, 2.0]), [4, 3, 2]),
            (torch.tensor([2.0, 2.0, 2.0]), [1, 3, 2]),
        ):
            controller.branch_ages = torch.tensor(ages)
            controller.collect_usage = lambda *a, **kw: (usage, 2)
            torch.manual_seed(19 + rank)
            controller._distributed_revival(trainer, module)
            controller.reset_adam_state(optimizer)
            assert not optimizer.state
            assert all(p.grad is None for p in backbone.parameters())
            assert parameter_ids == [id(p) for p in backbone.parameters()]
            assert controller.branch_ages.tolist() == (
                [0, 0, 3] if ages[0] == 4 else [2, 0, 3]
            )
            for value in backbone.state_dict().values():
                peer = value.clone()
                dist.broadcast(peer, src=0)
                assert torch.equal(value, peer)
        trainer.current_epoch = 3
        controller._era_iou_sum = 0.5
        controller.on_train_epoch_end(trainer, module)
        assert controller.accepted_epoch == 1
        for key, value in backbone.state_dict().items():
            assert torch.equal(value, accepted[key])
        # state_dict is deliberately callable on rank zero alone.
        if rank == 0:
            snapshot = controller.state_dict()
            assert snapshot["schema_version"] == 2
            assert len(snapshot["rank_rng_states"]) == 2
        dist.barrier()

        def broken(*a, **kw):
            raise ValueError("broken usage")

        controller.collect_usage = broken
        with pytest.raises(RuntimeError, match="broken usage"):
            controller._distributed_revival(trainer, module)

        def broken_decision(*a, **kw):
            raise ValueError("broken era decision")

        controller.finish_era = broken_decision
        with pytest.raises(RuntimeError, match="broken era decision"):
            controller.on_train_epoch_end(trainer, module)
        # Both ranks must leave the failed decision, without awaiting a timeout.
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_controller_real_collectives_and_failure_propagation(tmp_path):
    _launch(_controller_worker, tmp_path)
