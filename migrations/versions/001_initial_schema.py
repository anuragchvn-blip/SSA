"""Initial database schema for SSA engine.

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-01-17 07:59:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create TLE table
    op.create_table('tle',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('norad_id', sa.Integer(), nullable=False),
        sa.Column('classification', sa.String(length=1), nullable=True),
        sa.Column('launch_year', sa.Integer(), nullable=True),
        sa.Column('launch_number', sa.Integer(), nullable=True),
        sa.Column('launch_piece', sa.String(length=3), nullable=True),
        sa.Column('epoch_datetime', sa.DateTime(timezone=True), nullable=False),
        sa.Column('mean_motion_derivative', sa.Float(), nullable=True),
        sa.Column('mean_motion_sec_derivative', sa.Float(), nullable=True),
        sa.Column('bstar_drag_term', sa.Float(), nullable=True),
        sa.Column('element_set_number', sa.Integer(), nullable=True),
        sa.Column('inclination_degrees', sa.Float(), nullable=True),
        sa.Column('raan_degrees', sa.Float(), nullable=True),
        sa.Column('eccentricity', sa.Float(), nullable=True),
        sa.Column('argument_of_perigee_degrees', sa.Float(), nullable=True),
        sa.Column('mean_anomaly_degrees', sa.Float(), nullable=True),
        sa.Column('mean_motion_orbits_per_day', sa.Float(), nullable=True),
        sa.Column('revolution_number_at_epoch', sa.Integer(), nullable=True),
        sa.Column('tle_line1', sa.Text(), nullable=True),
        sa.Column('tle_line2', sa.Text(), nullable=True),
        sa.Column('epoch_julian_date', sa.Float(), nullable=True),
        sa.Column('line1_checksum', sa.Integer(), nullable=True),
        sa.Column('line2_checksum', sa.Integer(), nullable=True),
        sa.Column('source_url', sa.String(length=500), nullable=True),
        sa.Column('acquisition_timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('data_version', sa.String(length=50), nullable=True),
        sa.Column('is_valid', sa.Boolean(), nullable=True),
        sa.Column('validation_errors', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_tle_norad_epoch', 'tle', ['norad_id', 'epoch_datetime'], unique=False)
    op.create_index('idx_tle_epoch_datetime', 'tle', ['epoch_datetime'], unique=False)
    op.create_unique_constraint('unique_satellite_epoch', 'tle', ['norad_id', 'epoch_datetime'])

    # Create satellite_state table
    op.create_table('satellite_state',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tle_id', sa.Integer(), nullable=False),
        sa.Column('epoch_datetime', sa.DateTime(timezone=True), nullable=False),
        sa.Column('x_eci', sa.Float(), nullable=True),
        sa.Column('y_eci', sa.Float(), nullable=True),
        sa.Column('z_eci', sa.Float(), nullable=True),
        sa.Column('vx_eci', sa.Float(), nullable=True),
        sa.Column('vy_eci', sa.Float(), nullable=True),
        sa.Column('vz_eci', sa.Float(), nullable=True),
        sa.Column('covariance_rtn_flat', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('semi_major_axis_meters', sa.Float(), nullable=True),
        sa.Column('eccentricity', sa.Float(), nullable=True),
        sa.Column('inclination_degrees', sa.Float(), nullable=True),
        sa.Column('raan_degrees', sa.Float(), nullable=True),
        sa.Column('argument_of_perigee_degrees', sa.Float(), nullable=True),
        sa.Column('true_anomaly_degrees', sa.Float(), nullable=True),
        sa.Column('propagator_used', sa.String(length=50), nullable=True),
        sa.Column('force_models', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('step_size_seconds', sa.Float(), nullable=True),
        sa.Column('convergence_tolerance', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['tle_id'], ['tle.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_satellite_state_tle_epoch', 'satellite_state', ['tle_id', 'epoch_datetime'], unique=False)
    op.create_index('idx_satellite_state_epoch', 'satellite_state', ['epoch_datetime'], unique=False)

    # Create conjunction_event table
    op.create_table('conjunction_event',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('primary_norad_id', sa.Integer(), nullable=False),
        sa.Column('secondary_norad_id', sa.Integer(), nullable=False),
        sa.Column('tca_datetime', sa.DateTime(timezone=True), nullable=False),
        sa.Column('primary_x_eci', sa.Float(), nullable=True),
        sa.Column('primary_y_eci', sa.Float(), nullable=True),
        sa.Column('primary_z_eci', sa.Float(), nullable=True),
        sa.Column('secondary_x_eci', sa.Float(), nullable=True),
        sa.Column('secondary_y_eci', sa.Float(), nullable=True),
        sa.Column('secondary_z_eci', sa.Float(), nullable=True),
        sa.Column('miss_distance_meters', sa.Float(), nullable=True),
        sa.Column('relative_velocity_mps', sa.Float(), nullable=True),
        sa.Column('probability', sa.Float(), nullable=True),
        sa.Column('probability_method', sa.String(length=50), nullable=True),
        sa.Column('probability_confidence_lower', sa.Float(), nullable=True),
        sa.Column('probability_confidence_upper', sa.Float(), nullable=True),
        sa.Column('probability_samples', sa.Integer(), nullable=True),
        sa.Column('screening_threshold_km', sa.Float(), nullable=True),
        sa.Column('time_window_hours', sa.Float(), nullable=True),
        sa.Column('primary_object_name', sa.String(length=100), nullable=True),
        sa.Column('secondary_object_name', sa.String(length=100), nullable=True),
        sa.Column('primary_object_type', sa.String(length=50), nullable=True),
        sa.Column('secondary_object_type', sa.String(length=50), nullable=True),
        sa.Column('primary_radius_meters', sa.Float(), nullable=True),
        sa.Column('secondary_radius_meters', sa.Float(), nullable=True),
        sa.Column('alert_generated', sa.Boolean(), nullable=True),
        sa.Column('alert_threshold_exceeded', sa.Boolean(), nullable=True),
        sa.Column('alert_sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('analysis_version', sa.String(length=50), nullable=True),
        sa.Column('algorithm_parameters', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_conjunction_tca_datetime', 'conjunction_event', ['tca_datetime'], unique=False)
    op.create_index('idx_conjunction_probability', 'conjunction_event', ['probability'], unique=False)
    op.create_index('idx_conjunction_miss_distance', 'conjunction_event', ['miss_distance_meters'], unique=False)
    op.create_unique_constraint('unique_conjunction_tca', 'conjunction_event', ['primary_norad_id', 'secondary_norad_id', 'tca_datetime'])

    # Create maneuver_detection table
    op.create_table('maneuver_detection',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('norad_id', sa.Integer(), nullable=False),
        sa.Column('detection_datetime', sa.DateTime(timezone=True), nullable=False),
        sa.Column('maneuver_detected', sa.Boolean(), nullable=True),
        sa.Column('detection_confidence', sa.Float(), nullable=True),
        sa.Column('detection_method', sa.String(length=50), nullable=True),
        sa.Column('position_residual_magnitude', sa.Float(), nullable=True),
        sa.Column('velocity_residual_magnitude', sa.Float(), nullable=True),
        sa.Column('along_track_residual', sa.Float(), nullable=True),
        sa.Column('cross_track_residual', sa.Float(), nullable=True),
        sa.Column('radial_residual', sa.Float(), nullable=True),
        sa.Column('days_since_last_tle', sa.Float(), nullable=True),
        sa.Column('sma_change_meters', sa.Float(), nullable=True),
        sa.Column('eccentricity_change', sa.Float(), nullable=True),
        sa.Column('inclination_change_degrees', sa.Float(), nullable=True),
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.Column('feature_importance', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('shap_values', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('alert_generated', sa.Boolean(), nullable=True),
        sa.Column('alert_sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_maneuver_detection_norad_datetime', 'maneuver_detection', ['norad_id', 'detection_datetime'], unique=False)
    op.create_index('idx_maneuver_detection_datetime', 'maneuver_detection', ['detection_datetime'], unique=False)

    # Create alert_history table
    op.create_table('alert_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('alert_type', sa.String(length=50), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('recipient', sa.String(length=100), nullable=False),
        sa.Column('subject', sa.String(length=200), nullable=False),
        sa.Column('body', sa.Text(), nullable=False),
        sa.Column('payload', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('delivered', sa.Boolean(), nullable=True),
        sa.Column('delivery_confirmed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('delivery_attempts', sa.Integer(), nullable=True),
        sa.Column('conjunction_event_id', sa.Integer(), nullable=True),
        sa.Column('maneuver_detection_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['conjunction_event_id'], ['conjunction_event.id'], ),
        sa.ForeignKeyConstraint(['maneuver_detection_id'], ['maneuver_detection.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_alert_history_sent_at', 'alert_history', ['sent_at'], unique=False)
    op.create_index('idx_alert_history_alert_type', 'alert_history', ['alert_type'], unique=False)


def downgrade():
    op.drop_index('idx_alert_history_alert_type', table_name='alert_history')
    op.drop_index('idx_alert_history_sent_at', table_name='alert_history')
    op.drop_table('alert_history')

    op.drop_index('idx_maneuver_detection_datetime', table_name='maneuver_detection')
    op.drop_index('idx_maneuver_detection_norad_datetime', table_name='maneuver_detection')
    op.drop_table('maneuver_detection')

    op.drop_constraint('unique_conjunction_tca', 'conjunction_event', type_='unique')
    op.drop_index('idx_conjunction_miss_distance', table_name='conjunction_event')
    op.drop_index('idx_conjunction_probability', table_name='conjunction_event')
    op.drop_index('idx_conjunction_tca_datetime', table_name='conjunction_event')
    op.drop_table('conjunction_event')

    op.drop_index('idx_satellite_state_epoch', table_name='satellite_state')
    op.drop_index('idx_satellite_state_tle_epoch', table_name='satellite_state')
    op.drop_table('satellite_state')

    op.drop_constraint('unique_satellite_epoch', 'tle', type_='unique')
    op.drop_index('idx_tle_epoch_datetime', table_name='tle')
    op.drop_index('idx_tle_norad_epoch', table_name='tle')
    op.drop_table('tle')