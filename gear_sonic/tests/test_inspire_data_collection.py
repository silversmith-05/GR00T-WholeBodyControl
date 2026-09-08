"""Dataset and launch-argument tests. No cameras, Pico, deploy or hardware started."""
import json
import ast
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock

import numpy as np
import zmq
from gear_sonic.utils.data_collection.inspire_hand import (
    HAND_FEATURES, HAND_METADATA, HAND_SCHEMA_VERSION, ACTION_FIELDS, snapshot_to_frame, validate_snapshot,
    CLOSE_ANGLES, hand_metadata,
)
from gear_sonic.data.features_sonic_vla import (
    get_features_sonic_vla, get_modality_config_sonic_vla, get_g1_robot_model,
    get_wrist_camera_features,
)
from gear_sonic.data.exporter import Gr00tDataExporter, TypedLeRobotDataset
from gear_sonic.scripts.run_data_exporter import GrootDataCollector, SonicDataExporterConfig, validate_existing_dataset
from gear_sonic.scripts.launch_data_collection import DataCollectionLaunchConfig, hand_launch_arguments


def feedback(close_angles=CLOSE_ANGLES):
    now, mono = time.time(), time.monotonic()
    hands = []
    for target in (0, 1):
        hand = {key.removeprefix('hand.'): (False if dtype == 'bool' else 0)
                for key, (dtype, size, _) in HAND_FEATURES.items() if size == 2 and key not in ACTION_FIELDS}
        hand.update(target=target, angle=[501]*6, input_time=now, input_monotonic=mono,
                    angle_time=now, angle_monotonic=mono, write_time=now, write_status=2,
                    command_id=1, write_id=1, write_target=target, trigger=float(target),
                    thumb_rotation=339+target*10, write_thumb_rotation=339+target*10,
                    thumb_step=10+target*10, thumb_min=0, thumb_max=1000)
        hand.update(close_angles=list(close_angles), thumb_hold_rate=20.0,
                    angle_target=[*(close_angles if target else (1000,)*5),339+target*10])
        hand.update({key: True for key in ('connected','input_valid','target_valid','armed',
                                          'angle_valid','write_current','at_target_valid','thumb_target_valid')})
        hands.append(hand)
    return dict(schema_version=HAND_SCHEMA_VERSION, backend='inspire', model='RH56E2-T1',
                published_time=now, published_monotonic=mono, hands=hands)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = get_g1_robot_model()

    def collector(self, backend='inspire'):
        c = GrootDataCollector.__new__(GrootDataCollector)  # Never call hardware/subscriber constructors.
        c.hand_backend, c.robot_model = backend, self.robot
        c.latest_hand_msg = feedback()
        c._hand_episode_fault = False
        c._hand_fault_counts = [0, 0]
        c.latest_proprio_msg = dict(body_q=np.zeros(29), last_action=np.ones(29),
                                   workstation_receive_time=time.time())
        if backend == 'dex3':
            c.latest_proprio_msg.update(left_hand_q=np.ones(7)*.1, right_hand_q=np.ones(7)*.2,
                last_left_hand_action=np.ones(7)*.3, last_right_hand_action=np.ones(7)*.4)
        c.latest_image_msg = dict(images={'ego_view': np.zeros((480,640,3),dtype=np.uint8)},
                                  timestamps={'ego_view': time.time()})
        c.latest_sonic_msg = c.latest_planner_msg = None
        c.current_stream_mode = 1
        c._initial_yaw = None
        c._finalize_frame = lambda _: True
        c._log_latency_periodic = lambda _: None
        c.data_exporter = Mock()
        c.data_exporter.features = get_features_sonic_vla(self.robot, backend)
        return c

    def test_both_backends_produce_exact_declared_features(self):
        for backend in ('inspire', 'dex3'):
            c = self.collector(backend)
            c._add_data_frame_sonic(time.monotonic())
            frame = c.data_exporter.add_frame.call_args.args[0]
            features = c.data_exporter.features
            self.assertEqual(set(frame), set(features))
            for key, spec in features.items():
                self.assertEqual(tuple(frame[key].shape), tuple(spec['shape']), key)
                if spec['dtype'] != 'video':
                    self.assertEqual(str(frame[key].dtype), spec['dtype'], key)
            if backend == 'inspire':
                self.assertEqual(frame['action.hand'].tolist(), [0,1])
                self.assertEqual(frame['action.thumb_rotation'].tolist(), [339,349])
                self.assertEqual(frame['hand.thumb_step'].tolist(), [10,20])
                self.assertEqual(len(frame['observation.state']), 29)
                self.assertEqual(frame['action.wbc'].tolist(), [1]*29)
                self.assertNotIn('teleop.left_hand_joints', frame)
                self.assertEqual(frame['hand.angle_act'].tolist(), [501]*12)
                c._add_data_frame_sonic(time.monotonic())
                self.assertEqual(c.data_exporter.add_frame.call_args.args[0]['action.hand'].tolist(), [0,1])
            else:
                self.assertEqual(len(frame['observation.state']), 43)
                self.assertIn('teleop.left_hand_joints', frame)
                self.assertNotIn('action.hand', frame)
                self.assertNotIn('action.thumb_rotation', frame)

    def test_feedback_wire_receive_and_schema_failure(self):
        c = self.collector()
        snap = feedback()
        c._episode_state = Mock()
        c._episode_state.RECORDING = 1
        c._episode_state.get_state.return_value = 1
        c._sonic_zmq_socket = Mock()
        c._sonic_zmq_socket.recv.side_effect = [b"inspire_hand " + json.dumps(snap).encode(), zmq.Again()]
        c._poll_sonic_zmq_messages()
        self.assertEqual(c.latest_hand_msg, snap)
        snap["model"] = "Dex3"
        c._sonic_zmq_socket.recv.side_effect = [b"inspire_hand " + json.dumps(snap).encode(), zmq.Again()]
        c._poll_sonic_zmq_messages()
        self.assertIsNone(c.latest_hand_msg)
        self.assertTrue(c._hand_episode_fault)

    def test_training_modality_and_camera_resume_checks(self):
        modality = get_modality_config_sonic_vla(self.robot, 'inspire')
        self.assertEqual(modality['action']['hand'], dict(start=0, end=2, original_key='action.hand'))
        self.assertEqual(modality['action']['thumb_rotation'],
                         dict(start=0, end=2, original_key='action.thumb_rotation'))
        self.assertNotIn('left_hand', modality['state'])
        self.assertNotIn('left_hand_joints', modality['action'])
        for group in ('left_leg','right_leg','waist','left_arm','right_arm'):
            entry = modality['state'][group]
            indices = self.robot.get_joint_group_indices(group)
            body = self.robot.get_body_actuated_joint_indices()
            self.assertEqual(body[entry['start']:entry['end']], sorted(indices))
        features = get_features_sonic_vla(self.robot, 'inspire')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'meta').mkdir()
            (root/'meta/info.json').write_text(json.dumps(dict(features=features, fps=50,
                                                        script_config={'hand': HAND_METADATA})))
            validate_existing_dataset(root, features, 'inspire', 50)
            with self.assertRaises(ValueError):
                validate_existing_dataset(root, features, 'dex3', 50)
            with self.assertRaises(ValueError):
                validate_existing_dataset(root, {**features, **get_wrist_camera_features()}, 'inspire', 50)
            with self.assertRaises(ValueError):
                validate_existing_dataset(root, features, 'inspire', 20)

    def test_reversed_old_preset_metadata_and_wire_are_rejected(self):
        from gear_sonic.utils.data_collection.inspire_hand import validate_snapshot
        old_snapshot = feedback()
        old_snapshot['schema_version'] = 1
        with self.assertRaises(ValueError):
            validate_snapshot(old_snapshot)
        features = get_features_sonic_vla(self.robot, 'inspire')
        old_metadata = dict(HAND_METADATA, schema_version=1, preset_revision=1,
                            release=[1000,1000,1000,592,720,339],
                            close=[1000,1000,1000,1000,1000,339])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'meta').mkdir()
            (root/'meta/info.json').write_text(json.dumps(dict(features=features, fps=50,
                                                        script_config={'hand':old_metadata})))
            with self.assertRaisesRegex(ValueError, 'preset mapping differs'):
                validate_existing_dataset(root, features, 'inspire', 50)
        old_dataset = TypedLeRobotDataset.__new__(TypedLeRobotDataset)
        old_dataset.meta = Mock(info={'script_config': {'hand': old_metadata}})
        with self.assertRaisesRegex(ValueError, 'older/different hand action schema'):
            old_dataset.load_hf_dataset()
        self.assertEqual(HAND_METADATA['release'], [1000,1000,1000,1000,1000,339])
        self.assertEqual(HAND_METADATA['close'], [250,250,250,250,300,339])

    def test_fault_latches_even_when_next_feedback_recovers(self):
        c = self.collector()
        c._check_hand_faults(c.latest_hand_msg)
        self.assertFalse(c._hand_episode_fault)
        c.latest_hand_msg['hands'][0]['fault_count'] = 1
        c._check_hand_faults(c.latest_hand_msg)
        c._check_hand_faults(feedback())
        self.assertTrue(c._hand_episode_fault)
        c._save_episode()
        c.data_exporter.save_episode_as_discarded.assert_called_once()
        c.data_exporter.save_episode.assert_not_called()

    def test_pending_write_is_masked_but_not_a_failed_episode(self):
        snap = feedback()
        snap['hands'][0].update(command_id=2, write_current=False)
        f = snapshot_to_frame(snap, frame_time=time.time(), frame_monotonic=time.monotonic())
        self.assertFalse(f['hand.training_valid'][0])
        self.assertFalse(f['hand.episode_fault'][0])
        snap['hands'][0]['write_status'] = 3
        f = snapshot_to_frame(snap, frame_time=time.time(), frame_monotonic=time.monotonic())
        self.assertTrue(f['hand.episode_fault'][0])

    def test_launcher_argument_propagation_without_running_launcher(self):
        d,t,e = hand_launch_arguments(DataCollectionLaunchConfig(hand_backend='inspire', enable_hand_control=True,
            inspire_left_thumb_step=5, inspire_right_thumb_step=20,
            inspire_left_thumb_hold_rate=10.0, inspire_right_thumb_hold_rate=15.0,
            inspire_left_thumb_min=200, inspire_left_thumb_max=800,
            inspire_right_thumb_min=100, inspire_right_thumb_max=900))
        self.assertEqual(d, ['--disable-dex3-hands'])
        self.assertEqual(e, ['--hand-backend','inspire','--inspire-close-angles','250','250','250','250','300'])
        self.assertIn('--enable-hand-control',t)
        self.assertEqual(t[t.index('--inspire-left-ip')+1], '192.168.123.211')
        self.assertEqual(t[t.index('--inspire-right-ip')+1], '192.168.123.210')
        self.assertEqual(t[t.index('--inspire-left-thumb-hold-rate')+1], '10.0')
        self.assertEqual(t[t.index('--inspire-right-thumb-hold-rate')+1], '15.0')
        for side, step, low, high in [('left',5,200,800), ('right',20,100,900)]:
            for name, expected in [('step',step),('min',low),('max',high)]:
                self.assertEqual(t[t.index(f'--inspire-{side}-thumb-{name}')+1], str(expected))
        d,t,e = hand_launch_arguments(DataCollectionLaunchConfig())
        self.assertFalse(d)
        self.assertEqual(t, ['--hand-backend','dex3'])
        self.assertEqual(e, ['--hand-backend','dex3'])
        _,t,_ = hand_launch_arguments(DataCollectionLaunchConfig(hand_backend='inspire'))
        self.assertNotIn('--enable-hand-control',t)
        for side in ('left', 'right'):
            self.assertEqual(t[t.index(f'--inspire-{side}-thumb-hold-rate')+1], '50.0')
        with self.assertRaises(ValueError):
            hand_launch_arguments(DataCollectionLaunchConfig(hand_backend='inspire', inspire_left_thumb_step=0))
        # Parse the real CLI schemas, never run either entry point or import Pico runtime.
        import tyro
        config=tyro.cli(DataCollectionLaunchConfig,args=['--hand-backend','inspire',
            '--inspire-close-angles','230','240','250','260','310',
            '--inspire-left-thumb-hold-rate','10','--inspire-right-thumb-hold-rate','15'])
        _,teleop,exporter=hand_launch_arguments(config)
        recorded=tyro.cli(SonicDataExporterConfig,args=exporter)
        self.assertEqual(recorded.inspire_close_angles,(230,240,250,260,310))
        path=Path(__file__).resolve().parents[1]/'scripts/pico_manager_thread_server.py'
        module=ast.parse(path.read_text())
        entry=next(n for n in module.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")
        construction=[]
        for node in entry.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in node.targets):
                break
            construction.append(node)
        namespace={}
        exec(compile(ast.Module(body=construction,type_ignores=[]),str(path),'exec'),namespace)
        parsed=namespace['parser'].parse_args(['--manager',*teleop])
        self.assertEqual(parsed.inspire_close_angles,[230,240,250,260,310])
        self.assertEqual(parsed.inspire_left_thumb_hold_rate,10)
        self.assertEqual(parsed.inspire_right_thumb_hold_rate,15)
        defaults=namespace['parser'].parse_args(['--manager','--hand-backend','inspire'])
        self.assertEqual(defaults.inspire_left_thumb_hold_rate,50)
        self.assertEqual(defaults.inspire_right_thumb_hold_rate,50)
        call=next(n for n in ast.walk(entry) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Name) and n.func.id=='run_pico_manager')
        forwarded={kw.arg:ast.unparse(kw.value) for kw in call.keywords}
        for key in ('inspire_close_angles','inspire_left_thumb_hold_rate','inspire_right_thumb_hold_rate'):
            self.assertEqual(forwarded[key],f'args.{key}')

    def test_real_parquet_video_roundtrip_and_discard_filter(self):
        c = self.collector()
        custom = (230,240,250,260,310)
        c.inspire_close_angles = custom
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'dataset'
            exporter = Gr00tDataExporter.create(root, fps=50, features=c.data_exporter.features,
                modality_config=get_modality_config_sonic_vla(self.robot, 'inspire'),
                task='offline fake hand test', script_config={'hand':hand_metadata(custom)})
            c.data_exporter = exporter
            try:
                for _ in range(2):
                    c.latest_hand_msg = feedback(custom)
                    c._add_data_frame_sonic(time.monotonic())
                c._save_episode()
                for _ in range(2):
                    c.latest_hand_msg = feedback(custom)
                    c.latest_hand_msg['hands'][0]['write_status'] = 3
                    c._add_data_frame_sonic(time.monotonic())
                c._save_episode()
            finally:
                exporter.stop_video_writers()
            info = json.loads((root/'meta/info.json').read_text())
            self.assertEqual(info['discarded_episode_indices'], [1])
            self.assertEqual(info['features']['action.hand']['shape'], [2])
            self.assertEqual(info['features']['action.thumb_rotation']['shape'], [2])
            self.assertEqual(info['script_config']['hand']['close'], [230,240,250,260,310,339])
            import pyarrow.parquet as pq
            table = pq.read_table(root/'data/chunk-000/episode_000001.parquet')
            self.assertEqual(table['hand.write_status'].to_pylist()[0], [3,2])
            # Older schema-4 recordings remain readable/resumable: actual rates
            # are per-frame, so the old descriptive default is not a schema change.
            info['script_config']['hand']['thumb_rotation']['hold_rate'] = (
                HAND_METADATA['thumb_rotation']['hold_rate'].replace('; at most', '; default 20, at most'))
            (root/'meta/info.json').write_text(json.dumps(info))
            validate_existing_dataset(root, c.data_exporter.features, 'inspire', 50,
                                      expected_hand=hand_metadata(custom))
            dataset = TypedLeRobotDataset(repo_id='tmp/tmp_dataset', root=root, load_video=False, download_videos=False)
            self.assertEqual(dataset.episodes, [0])
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.hf_dataset[0]['action.hand'].tolist(), [0,1])
            self.assertEqual(dataset.hf_dataset[0]['action.thumb_rotation'].tolist(), [339,349])
            self.assertEqual(dataset.hf_dataset[0]['hand.thumb_step'].tolist(), [10,20])
            self.assertEqual(dataset.hf_dataset[0]['hand.thumb_hold_rate'].tolist(), [20,20])
            self.assertEqual(dataset.hf_dataset[0]['hand.close_angles'].tolist(), list(custom)*2)
            self.assertEqual(dataset.hf_dataset[0]['hand.angle_target'].tolist(), [1000]*5+[339]+list(custom)+[349])
            self.assertTrue(dataset.hf_dataset[0]['hand.training_valid'].item())

    def test_thumb_validity_and_write_result_are_required_for_training(self):
        for key, value in [('thumb_target_valid',False), ('connected',False)]:
            snap = feedback()
            snap['hands'][0][key] = value
            frame = snapshot_to_frame(snap,frame_time=time.time(),frame_monotonic=time.monotonic())
            self.assertFalse(frame['hand.training_valid'][0])
            self.assertTrue(frame['hand.episode_fault'][0])
        snap = feedback()
        snap['hands'][0]['write_thumb_rotation'] -= 10
        frame = snapshot_to_frame(snap,frame_time=time.time(),frame_monotonic=time.monotonic())
        self.assertFalse(frame['hand.write_current'][0])
        self.assertFalse(frame['hand.training_valid'][0])
        self.assertFalse(frame['hand.episode_fault'][0])
        snap = feedback()
        for hand in snap['hands']:
            hand['angle_monotonic'] -= 1
        frame = snapshot_to_frame(snap,frame_time=time.time(),frame_monotonic=time.monotonic())
        self.assertFalse(frame['hand.at_target_valid'].any())
        self.assertTrue(frame['hand.episode_fault'][0])

    def test_thumb_schema_rejects_binary_only_or_bad_rotation(self):
        for key, value in [('thumb_rotation',1001), ('thumb_rotation',float('nan')),
                           ('thumb_rotation',339.5),('thumb_step',0),('thumb_min',1001)]:
            snap = feedback()
            snap['hands'][0][key] = value
            with self.assertRaises(ValueError):
                validate_snapshot(snap)
        snap = feedback()
        snap['schema_version'] = 2
        with self.assertRaises(ValueError):
            validate_snapshot(snap)
        old_metadata = dict(HAND_METADATA, schema_version=2)
        old_metadata.pop('thumb_rotation')
        features = get_features_sonic_vla(self.robot, 'inspire')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'meta').mkdir()
            (root/'meta/info.json').write_text(json.dumps(dict(features=features, fps=50,
                                                        script_config={'hand':old_metadata})))
            with self.assertRaises(ValueError):
                validate_existing_dataset(root, features, 'inspire', 50)

    def test_five_finger_configuration_resume_and_wire_mismatch_are_rejected(self):
        custom=(230,240,250,260,310)
        metadata=hand_metadata(custom)
        features=get_features_sonic_vla(self.robot,'inspire')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'meta').mkdir()
            (root/'meta/info.json').write_text(json.dumps(dict(features=features,fps=50,script_config={'hand':metadata})))
            validate_existing_dataset(root,features,'inspire',50,metadata)
            with self.assertRaises(ValueError):
                validate_existing_dataset(root,features,'inspire',50,HAND_METADATA)
        c=self.collector()
        c._episode_state=Mock(RECORDING=1)
        c._episode_state.get_state.return_value=1
        c._sonic_zmq_socket=Mock()
        c._sonic_zmq_socket.recv.side_effect=[b'inspire_hand '+json.dumps(feedback(custom)).encode(),zmq.Again()]
        c._poll_sonic_zmq_messages()
        self.assertIsNone(c.latest_hand_msg)
        self.assertTrue(c._hand_episode_fault)
        for version in (1,2,3):
            snap=feedback()
            snap['schema_version']=version
            with self.assertRaises(ValueError):
                validate_snapshot(snap)

    def test_hold_preparation_cancellation_is_not_false_motion_failure(self):
        snap=feedback()
        snap['hands'][0]['thumb_hold_cancel_count']=1
        frame=snapshot_to_frame(snap,frame_time=time.time(),frame_monotonic=time.monotonic())
        self.assertTrue(frame['hand.training_valid'][0])
        self.assertFalse(frame['hand.episode_fault'][0])
        self.assertEqual(frame['hand.thumb_hold_cancel_count'].tolist(),[1,0])
        snap['hands'][0].update(write_status=1,write_thumb_rotation=341)
        frame=snapshot_to_frame(snap,frame_time=time.time(),frame_monotonic=time.monotonic())
        self.assertFalse(frame['hand.training_valid'][0])
        self.assertFalse(frame['hand.episode_fault'][0])
        snap['hands'][0]['write_status']=3
        frame=snapshot_to_frame(snap,frame_time=time.time(),frame_monotonic=time.monotonic())
        self.assertTrue(frame['hand.episode_fault'][0])


if __name__ == '__main__':
    unittest.main()
