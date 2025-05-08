"""
Utility functions for InfluxDB 3 integration using influxdb3-python client.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import InfluxDB3 client
try:
    # The package name is influxdb3-python but the import is influxdb_client_3
    from influxdb_client_3 import InfluxDBClient3, Point, WriteOptions
    logger.info("Successfully imported influxdb_client_3")
    INFLUXDB_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import influxdb_client_3: {str(e)}")
    logger.warning("InfluxDB functionality will be disabled. Install with 'pip install influxdb3-python'")
    INFLUXDB_AVAILABLE = False
    
    # Create dummy classes for type checking when InfluxDB is not available
    class DummyInfluxDBClient3:
        def __init__(self, *args, **kwargs):
            self.host = kwargs.get('host', 'dummy_host')
            self.database = kwargs.get('database', 'dummy_db')
            self.org = kwargs.get('org', '')
            
        def version(self):
            """Dummy version method"""
            return "0.0.0"
            
        def write(self, data=None, *args, **kwargs):
            """Dummy write method"""
            logger.warning("InfluxDB write called on dummy client")
            if data is not None:
                logger.info(f"Would have written data of type {type(data).__name__}")
            return None
            
        def query(self, query="", language="sql", *args, **kwargs):
            """Dummy query method"""
            logger.warning(f"InfluxDB query called on dummy client: {query}")
            # Return an empty DataFrame that has basic properties needed
            return pd.DataFrame()
            
    class DummyPoint:
        def __init__(self, measurement=None):
            self.measurement = measurement
            self.tags = {}
            self.fields = {}
            self.timestamp = None
        
        def tag(self, key, value):
            """Add a tag to the point"""
            self.tags[key] = value
            return self
            
        def field(self, key, value):
            """Add a field to the point"""
            self.fields[key] = value
            return self
            
        def time(self, timestamp):
            """Set the time for the point"""
            self.timestamp = timestamp
            return self
    
    # Use dummy classes if real ones are not available
    InfluxDBClient3 = DummyInfluxDBClient3
    Point = DummyPoint
class InfluxDBHandler:
    """
    Handler for InfluxDB 3 operations using the influxdb3-python client.
    """
    
    def __init__(self, host=None, token=None, database=None, org=""):
        """
        Initialize the InfluxDB 3 handler.
        
        Parameters:
        -----------
        host : str
            InfluxDB host URL
        token : str
            Authentication token
        database : str
            Database name
        org : str
            Organization name (optional)
        """
        # Try to get values from environment variables if not provided
        self.host = host or os.environ.get("INFLUXDB_HOST", "http://localhost:8181")
        self.token = token or os.environ.get("INFLUXDB_TOKEN", "")
        self.database = database or os.environ.get("INFLUXDB_DATABASE", "timeseries")
        self.org = org or os.environ.get("INFLUXDB_ORG", "")
        
        self.client = None
        self.connected = False
    
    def connect(self):
        """
        Connect to InfluxDB.
        
        Returns:
        --------
        bool
            True if connection is successful, False otherwise
        """
        if not INFLUXDB_AVAILABLE:
            logger.warning("InfluxDB client not available. Install with 'pip install influxdb3-python'")
            self.connected = False
            return False
            
        try:
            # Create InfluxDB client
            self.client = InfluxDBClient3(
                token=self.token,
                host=self.host,
                database=self.database,
                org=self.org
            )
            
            # Test connection by listing measurements (simple query)
            try:
                test_query = "SHOW MEASUREMENTS"
                self.client.query(query=test_query, language="sql")
                self.connected = True
                logger.info(f"Successfully connected to InfluxDB at {self.host}")
                return True
            except Exception as e:
                logger.error(f"Connection to InfluxDB established but query failed: {str(e)}")
                self.connected = True  # Still mark as connected if server responds but query fails
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {str(e)}")
            self.connected = False
            return False
    
    def write_dataframe(self, df, measurement_name, tag_columns=None):
        """
        Write a DataFrame to InfluxDB.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to write
        measurement_name : str
            Name of the measurement
        tag_columns : list
            List of column names to use as tags
            
        Returns:
        --------
        bool
            True if write is successful, False otherwise
        """
        # Check for valid DataFrame
        if df is None or len(df) == 0:
            logger.warning("Cannot write empty DataFrame to InfluxDB")
            return False
            
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Expected pd.DataFrame, got {type(df).__name__}")
            return False
            
        # Validate measurement name
        if not measurement_name:
            logger.error("No measurement name provided")
            return False
            
        # Check connection status
        if not self.connected or self.client is None:
            logger.info("Not connected to InfluxDB, attempting to connect")
            if not self.connect():
                logger.error("Failed to connect to InfluxDB")
                return False
        
        try:
            # Make sure DataFrame has timestamps as index if not already
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    logger.info("Setting 'timestamp' column as index")
                    df = df.set_index('timestamp')
                else:
                    logger.warning("DataFrame has no timestamp index or column")
                    
            # Write DataFrame to InfluxDB
            logger.info(f"Writing {len(df)} records to measurement '{measurement_name}'")
            
            # Use tag columns if provided
            if tag_columns:
                logger.info(f"Using tag columns: {tag_columns}")
                
            self.client.write(
                df, 
                data_frame_measurement_name=measurement_name,
                data_frame_tag_columns=tag_columns
            )
            
            logger.info(f"Successfully wrote {len(df)} records to measurement '{measurement_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write DataFrame to InfluxDB: {str(e)}")
            return False
    
    def query_data(self, sql_query=None, measurement=None, start_time=None, end_time=None, limit=None):
        """
        Query data from InfluxDB.
        
        Parameters:
        -----------
        sql_query : str
            SQL query to execute (if provided, other parameters are ignored)
        measurement : str
            Name of the measurement to query
        start_time : datetime or str
            Start time for the query
        end_time : datetime or str
            End time for the query
        limit : int
            Maximum number of records to return
            
        Returns:
        --------
        pd.DataFrame
            Query results as a DataFrame, or empty DataFrame if query fails
        """
        # Default to empty DataFrame if query fails
        empty_result = pd.DataFrame()
        
        # Check connection status
        if not self.connected or self.client is None:
            if not self.connect():
                logger.warning("Query failed: Not connected to InfluxDB")
                return empty_result
        
        try:
            # Determine query to execute
            if sql_query:
                # Use provided SQL query directly
                query = sql_query
            else:
                # Build a query from parameters
                if not measurement:
                    logger.error("No measurement specified for query")
                    return empty_result
                    
                query_parts = [f"SELECT * FROM {measurement}"]
                
                # Add time filters
                if start_time:
                    start_str = start_time if isinstance(start_time, str) else start_time.isoformat()
                    query_parts.append(f"WHERE time >= '{start_str}'")
                    
                    if end_time:
                        end_str = end_time if isinstance(end_time, str) else end_time.isoformat()
                        query_parts.append(f"AND time <= '{end_str}'")
                elif end_time:
                    end_str = end_time if isinstance(end_time, str) else end_time.isoformat()
                    query_parts.append(f"WHERE time <= '{end_str}'")
                
                # Add order and limit
                query_parts.append("ORDER BY time ASC")
                
                if limit and isinstance(limit, int) and limit > 0:
                    query_parts.append(f"LIMIT {limit}")
                
                query = " ".join(query_parts)
            
            # Execute the query
            logger.info(f"Executing query: {query}")
            result = self.client.query(query=query, language="sql")
            
            # Check if result is None or empty
            if result is None:
                logger.warning("Query returned None")
                return empty_result
            
            # Log success and return result
            record_count = len(result) if hasattr(result, '__len__') else "unknown"
            logger.info(f"Successfully executed query, retrieved {record_count} records")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute query: {str(e)}")
            return empty_result

    def write_point(self, measurement, tags=None, fields=None, time=None):
        """
        Write a single point to InfluxDB.
        
        Parameters:
        -----------
        measurement : str
            Name of the measurement
        tags : dict
            Dictionary of tag keys and values
        fields : dict
            Dictionary of field keys and values
        time : datetime
            Timestamp for the point
            
        Returns:
        --------
        bool
            True if write is successful, False otherwise
        """
        # Validate measurement name
        if not measurement:
            logger.error("No measurement name provided for write_point")
            return False
            
        # Require at least one field
        if not fields or not isinstance(fields, dict) or len(fields) == 0:
            logger.error("No fields provided for write_point")
            return False
            
        # Check connection status
        if not self.connected or self.client is None:
            logger.info("Not connected to InfluxDB, attempting to connect")
            if not self.connect():
                logger.error("Failed to connect to InfluxDB")
                return False
        
        try:
            # Create point
            point = Point(measurement)
            
            # Add tags if provided
            if tags and isinstance(tags, dict):
                for key, value in tags.items():
                    if value is not None:  # Skip None values
                        point.tag(key, str(value))  # Convert to string for safety
            
            # Add fields (required)
            field_count = 0
            for key, value in fields.items():
                if value is not None:  # Skip None values
                    point.field(key, value)
                    field_count += 1
                    
            # Check if we have at least one field
            if field_count == 0:
                logger.error("No valid fields (all were None)")
                return False
            
            # Add time if provided
            if time:
                point.time(time)
            
            # Write point
            self.client.write(point)
            
            # Log success
            tag_str = f" with {len(tags) if tags else 0} tags" if tags else ""
            field_str = f" and {field_count} fields"
            time_str = " and timestamp" if time else ""
            logger.info(f"Successfully wrote point to measurement '{measurement}'{tag_str}{field_str}{time_str}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write point to InfluxDB: {str(e)}")
            return False
            
    def check_measurement_exists(self, measurement):
        """
        Check if a measurement exists in InfluxDB.
        
        Parameters:
        -----------
        measurement : str
            Name of the measurement to check
            
        Returns:
        --------
        bool
            True if measurement exists, False otherwise
        """
        if not self.connected or self.client is None:
            if not self.connect():
                return False
                
        try:
            # Query for measurements
            result = self.query_data(sql_query="SHOW MEASUREMENTS")
            
            # Check if result contains the measurement
            if result is None or result.empty:
                return False
                
            # Convert to list if it's a DataFrame
            measurements = result.values.flatten().tolist() if hasattr(result, 'values') else []
            
            # Check if measurement exists
            return measurement in measurements
            
        except Exception as e:
            logger.error(f"Failed to check if measurement exists: {str(e)}")
            return False
            
    def create_required_measurements(self, measurements=None):
        """
        Create required measurements if they don't exist by writing a dummy point.
        
        Parameters:
        -----------
        measurements : list
            List of measurement names to create, if None, creates default measurements
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if measurements is None:
            measurements = [
                "raw_data",              # Original time series data
                "drifted_data",          # Data with injected drift
                "model_metrics",         # Model performance metrics
                "drift_metrics",         # Drift detection metrics
                "model_events",          # Model training/retraining events
                "forecasts"              # Model forecasts
            ]
            
        if not self.connected or self.client is None:
            if not self.connect():
                logger.error("Failed to connect to InfluxDB")
                return False
                
        success = True
        for measurement in measurements:
            # Check if measurement exists
            if not self.check_measurement_exists(measurement):
                # Create measurement with dummy point
                logger.info(f"Creating measurement '{measurement}'")
                
                # Get current time
                now = datetime.now()
                
                # Write dummy point with initialization tag
                result = self.write_point(
                    measurement=measurement,
                    tags={
                        "type": "initialization",
                        "created_by": "InfluxDBHandler"
                    },
                    fields={
                        "initialized": True,
                        "timestamp_str": now.isoformat()
                    },
                    time=now
                )
                
                if not result:
                    logger.error(f"Failed to create measurement '{measurement}'")
                    success = False
                else:
                    logger.info(f"Successfully created measurement '{measurement}'")
                    
        return success