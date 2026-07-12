import bpy
from mathutils import Matrix, Vector
from bl_ui.utils import PresetPanel
from bl_operators.presets import AddPresetBase

bl_info = {
    # required
    'name': 'IK/FK Snapping',
    'blender': (3, 1, 0),
    'category': 'Animation',
    # optional
    'version': (2, 0, 0),
    'author': 'Byron Mallett',
    'description': 'Custom rig FK/IK snapping tools',
}


def poll_armature_object(self, obj):
    return obj.type == 'ARMATURE'


class FKIKLimbSettings(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name='Limb name', default='Limb')
    armature: bpy.props.PointerProperty(
        name='Armature',
        type=bpy.types.Object,
        poll=poll_armature_object,
    )
    fk_upper: bpy.props.StringProperty(name='FK upper')
    fk_lower: bpy.props.StringProperty(name='FK lower')
    fk_end: bpy.props.StringProperty(name='FK hand/foot')
    ik_upper: bpy.props.StringProperty(name='IK upper')
    ik_lower: bpy.props.StringProperty(name='IK lower')
    ik_end: bpy.props.StringProperty(name='IK hand/foot')
    ik_target: bpy.props.StringProperty(name='IK target (control)')
    ik_pole: bpy.props.StringProperty(name='IK pole')


def get_active_limb(scene):
    if 0 <= scene.fkik_limb_index < len(scene.fkik_limbs):
        return scene.fkik_limbs[scene.fkik_limb_index]
    return None


def resolve_bones(limb, fields):
    """Look up the limb's pose bones for the given fields.

    Returns (bones dict keyed by field, list of human-readable missing field names).
    """
    bones = {}
    missing = []
    for field in fields:
        bone_name = getattr(limb, field)
        pbone = limb.armature.pose.bones.get(bone_name) if bone_name else None
        if pbone is None:
            missing.append(limb.bl_rna.properties[field].name)
        else:
            bones[field] = pbone
    return bones, missing


def keyframe_snapped_bone(pbone, frame):
    pbone.keyframe_insert('location', frame=frame)
    if pbone.rotation_mode == 'QUATERNION':
        pbone.keyframe_insert('rotation_quaternion', frame=frame)
    elif pbone.rotation_mode == 'AXIS_ANGLE':
        pbone.keyframe_insert('rotation_axis_angle', frame=frame)
    else:
        pbone.keyframe_insert('rotation_euler', frame=frame)


class SnapOperatorBase:
    required_fields = ()
    keyed_fields = ()

    @classmethod
    def poll(cls, context):
        return get_active_limb(context.scene) is not None

    def execute(self, context):
        scene = context.scene
        limb = get_active_limb(scene)
        if limb is None:
            self.report({'ERROR'}, 'No limb selected')
            return {'CANCELLED'}
        if limb.armature is None or limb.armature.type != 'ARMATURE':
            self.report({'ERROR'}, "Limb '%s' has no armature assigned" % limb.name)
            return {'CANCELLED'}

        bones, missing = resolve_bones(limb, self.required_fields)
        if missing:
            self.report(
                {'ERROR'},
                "Limb '%s': assign valid bones for: %s" % (limb.name, ', '.join(missing)),
            )
            return {'CANCELLED'}

        if scene.fkik_use_frame_range:
            if scene.fkik_start_frame > scene.fkik_end_frame:
                self.report({'ERROR'}, 'Start frame must not be after end frame')
                return {'CANCELLED'}
            original_frame = scene.frame_current
            for frame in range(scene.fkik_start_frame, scene.fkik_end_frame + 1):
                scene.frame_set(frame)
                self.snap(context, bones)
                for field in self.keyed_fields:
                    keyframe_snapped_bone(bones[field], frame)
            scene.frame_set(original_frame)
        else:
            # No frame_set here: it re-evaluates animation, which would wipe
            # both the user's unkeyed pose and the snap result on keyed bones
            self.snap(context, bones)
            if scene.tool_settings.use_keyframe_insert_auto:
                for field in self.keyed_fields:
                    keyframe_snapped_bone(bones[field], scene.frame_current)

        self.report({'INFO'}, "%s: limb '%s'" % (self.bl_label, limb.name))
        return {'FINISHED'}


class FKIK_OT_snap_ik_to_fk(SnapOperatorBase, bpy.types.Operator):
    bl_idname = 'fkik.snap_ik_to_fk'
    bl_label = 'Snap IK to FK'
    bl_description = 'Move the IK target and pole so the IK chain matches the current FK pose'
    bl_options = {'REGISTER', 'UNDO'}

    required_fields = ('fk_upper', 'fk_lower', 'fk_end', 'ik_target', 'ik_pole')
    keyed_fields = ('ik_target', 'ik_pole')

    def snap(self, context, bones):
        fk_upper = bones['fk_upper']
        fk_lower = bones['fk_lower']
        fk_end = bones['fk_end']
        ik_target = bones['ik_target']
        ik_pole = bones['ik_pole']

        # Set IK target matrix relative to the original FK end bone in armature space
        target_offset = fk_end.bone.matrix_local.inverted() @ ik_target.bone.matrix_local
        ik_target.matrix = fk_end.matrix @ target_offset
        context.view_layer.update()

        # Place the pole on the FK bend plane: perpendicular from the
        # root->tip chord through the elbow/knee, scaled to the chain length
        upper_head = fk_upper.matrix.translation
        elbow = fk_lower.matrix.translation
        chain_tip = elbow + fk_lower.vector
        chord = chain_tip - upper_head
        elbow_offset = elbow - upper_head
        if chord.length_squared > 1e-12:
            pole_dir = elbow_offset - elbow_offset.project(chord)
        else:
            pole_dir = Vector()
        if pole_dir.length_squared < 1e-12:
            # Limb is straight: keep the pole on its current side of the chain
            pole_dir = ik_pole.matrix.translation - elbow
        if pole_dir.length_squared < 1e-12:
            return

        pole_loc = elbow + pole_dir.normalized() * (fk_upper.length + fk_lower.length)
        ik_pole.matrix = Matrix.LocRotScale(pole_loc, ik_pole.matrix.to_quaternion(), None)
        context.view_layer.update()


class FKIK_OT_snap_fk_to_ik(SnapOperatorBase, bpy.types.Operator):
    bl_idname = 'fkik.snap_fk_to_ik'
    bl_label = 'Snap FK to IK'
    bl_description = 'Copy the IK chain pose onto the FK bones'
    bl_options = {'REGISTER', 'UNDO'}

    required_fields = ('ik_upper', 'ik_lower', 'ik_end', 'fk_upper', 'fk_lower', 'fk_end')
    keyed_fields = ('fk_upper', 'fk_lower', 'fk_end')

    def snap(self, context, bones):
        bones['fk_upper'].matrix = bones['ik_upper'].matrix
        context.view_layer.update()

        bones['fk_lower'].matrix = bones['ik_lower'].matrix
        context.view_layer.update()

        # Set FK end matrix relative to the original IK end bone in armature space
        end_offset = bones['ik_end'].bone.matrix_local.inverted() @ bones['fk_end'].bone.matrix_local
        bones['fk_end'].matrix = bones['ik_end'].matrix @ end_offset
        context.view_layer.update()


class FKIK_OT_limb_add(bpy.types.Operator):
    bl_idname = 'fkik.limb_add'
    bl_label = 'Add Limb'
    bl_description = 'Add a new limb configuration'

    def execute(self, context):
        scene = context.scene
        limb = scene.fkik_limbs.add()
        limb.name = 'Limb %d' % len(scene.fkik_limbs)
        ob = context.active_object
        if ob is not None and ob.type == 'ARMATURE':
            limb.armature = ob
        scene.fkik_limb_index = len(scene.fkik_limbs) - 1
        return {'FINISHED'}


class FKIK_OT_limb_remove(bpy.types.Operator):
    bl_idname = 'fkik.limb_remove'
    bl_label = 'Remove Limb'
    bl_description = 'Remove the selected limb configuration'

    @classmethod
    def poll(cls, context):
        return get_active_limb(context.scene) is not None

    def execute(self, context):
        scene = context.scene
        scene.fkik_limbs.remove(scene.fkik_limb_index)
        scene.fkik_limb_index = min(scene.fkik_limb_index, len(scene.fkik_limbs) - 1)
        return {'FINISHED'}


class FKIK_MT_limb_presets(bpy.types.Menu):
    bl_label = 'Limb Presets'
    bl_idname = 'FKIK_MT_limb_presets'
    preset_subdir = 'object/FKIKSnap_presets'
    preset_operator = 'script.execute_preset'
    draw = bpy.types.Menu.draw_preset


class FKIK_PT_presets(PresetPanel, bpy.types.Panel):
    bl_label = 'Limb Presets'
    preset_subdir = 'object/FKIKSnap_presets'
    preset_operator = 'script.execute_preset'
    preset_add_operator = 'fkik.add_limb_preset'


class FKIK_OT_add_limb_preset(AddPresetBase, bpy.types.Operator):
    bl_idname = 'fkik.add_limb_preset'
    bl_label = 'Add Limb Preset'
    preset_menu = 'FKIK_MT_limb_presets'

    # Presets store only bone names (not the armature), so a preset made on
    # one rig can be applied to the active limb of any rig with the same naming
    preset_defines = [
        'limb = bpy.context.scene.fkik_limbs[bpy.context.scene.fkik_limb_index]',
    ]

    preset_values = [
        'limb.name',
        'limb.fk_upper',
        'limb.fk_lower',
        'limb.fk_end',
        'limb.ik_upper',
        'limb.ik_lower',
        'limb.ik_end',
        'limb.ik_target',
        'limb.ik_pole',
    ]

    preset_subdir = 'object/FKIKSnap_presets'


class FKIK_UL_limbs(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        row = layout.row(align=True)
        row.prop(item, 'name', text='', emboss=False)
        if item.armature is not None:
            row.label(text=item.armature.name, icon='ARMATURE_DATA')


class FKIKSnapPanel(bpy.types.Panel):
    bl_idname = 'VIEW3D_PT_fk_to_ik_snap'
    bl_label = 'FK/IK snapping'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'FK/IK Snap'

    def draw_header_preset(self, _context):
        FKIK_PT_presets.draw_panel_header(self.layout)

    def draw(self, context):
        scene = context.scene
        col = self.layout.column()
        col.prop(scene, 'fkik_use_frame_range')
        row = col.row(align=True)
        row.enabled = scene.fkik_use_frame_range
        row.prop(scene, 'fkik_start_frame')
        row.prop(scene, 'fkik_end_frame')

        limb = get_active_limb(scene)
        if limb is None:
            col.label(text='Add a limb in the FK/IK bones panel', icon='INFO')
            return

        col.separator()
        col.label(text=limb.name)
        col.operator('fkik.snap_ik_to_fk', text='Snap IK to FK')
        col.operator('fkik.snap_fk_to_ik', text='Snap FK to IK')


class FKIKMappingPanel(bpy.types.Panel):
    bl_idname = 'VIEW3D_PT_fk_to_ik_mapping'
    bl_label = 'FK/IK bones'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'FK/IK Snap'

    def draw_header_preset(self, _context):
        FKIK_PT_presets.draw_panel_header(self.layout)

    def draw(self, context):
        scene = context.scene
        layout = self.layout

        row = layout.row()
        row.template_list('FKIK_UL_limbs', '', scene, 'fkik_limbs', scene, 'fkik_limb_index')
        col = row.column(align=True)
        col.operator('fkik.limb_add', icon='ADD', text='')
        col.operator('fkik.limb_remove', icon='REMOVE', text='')

        limb = get_active_limb(scene)
        if limb is None:
            return

        col = layout.column()
        col.prop(limb, 'armature')
        if limb.armature is None or limb.armature.type != 'ARMATURE':
            return

        arma = limb.armature.data
        col.separator()
        col.label(text='FK chain:')
        col.prop_search(limb, 'fk_upper', arma, 'bones')
        col.prop_search(limb, 'fk_lower', arma, 'bones')
        col.prop_search(limb, 'fk_end', arma, 'bones')
        col.separator()
        col.label(text='IK chain:')
        col.prop_search(limb, 'ik_upper', arma, 'bones')
        col.prop_search(limb, 'ik_lower', arma, 'bones')
        col.prop_search(limb, 'ik_end', arma, 'bones')
        col.separator()
        col.label(text='IK controls:')
        col.prop_search(limb, 'ik_target', arma, 'bones')
        col.prop_search(limb, 'ik_pole', arma, 'bones')


CLASSES = [
    FKIKLimbSettings,
    FKIK_UL_limbs,
    FKIK_OT_limb_add,
    FKIK_OT_limb_remove,
    FKIK_OT_snap_ik_to_fk,
    FKIK_OT_snap_fk_to_ik,
    FKIK_OT_add_limb_preset,
    FKIK_PT_presets,
    FKIK_MT_limb_presets,
    FKIKSnapPanel,
    FKIKMappingPanel,
]

SCENE_PROPS = {
    'fkik_limbs': bpy.props.CollectionProperty(type=FKIKLimbSettings),
    'fkik_limb_index': bpy.props.IntProperty(name='Active limb', default=0),
    'fkik_use_frame_range': bpy.props.BoolProperty(name='Key across frame range', default=False),
    'fkik_start_frame': bpy.props.IntProperty(name='Start frame', default=1),
    'fkik_end_frame': bpy.props.IntProperty(name='End frame', default=1),
}


def register():
    for klass in CLASSES:
        bpy.utils.register_class(klass)

    for prop_name, prop_value in SCENE_PROPS.items():
        setattr(bpy.types.Scene, prop_name, prop_value)


def unregister():
    for prop_name in SCENE_PROPS:
        delattr(bpy.types.Scene, prop_name)

    for klass in reversed(CLASSES):
        bpy.utils.unregister_class(klass)


if __name__ == '__main__':
    register()
