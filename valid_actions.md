# Valid Actions for Mini-Games

#### MoveToBeacon

```python
0/no_op                       ()
1/move_camera                 (1/minimap [64, 64])
2/select_point                (6/select_point_act [4]; 0/screen [84, 84])
3/select_rect                 (7/select_add [2]; 0/screen [84, 84]; 2/screen2 [84, 84])
4/select_control_group        (4/control_group_act [5]; 5/control_group_id [10])
7/select_army                 (7/select_add [2])
12/Attack_screen              (3/queued [2]; 0/screen [84, 84])
13/Attack_minimap             (3/queued [2]; 1/minimap [64, 64])
274/HoldPosition_quick        (3/queued [2])
331/Move_screen               (3/queued [2]; 0/screen [84, 84])
332/Move_minimap              (3/queued [2]; 1/minimap [64, 64])
333/Patrol_screen             (3/queued [2]; 0/screen [84, 84])
334/Patrol_minimap            (3/queued [2]; 1/minimap [64, 64])
453/Stop_quick                (3/queued [2])
```

