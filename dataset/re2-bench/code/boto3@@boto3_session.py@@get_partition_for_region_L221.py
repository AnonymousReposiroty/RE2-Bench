import os
import botocore.session
from botocore.exceptions import (
    DataNotFoundError,
    NoCredentialsError,
    UnknownServiceError,
)
import boto3
import boto3.utils
from .resources.factory import ResourceFactory

class Session:
    """
    A session stores configuration state and allows you to create service
    clients and resources.

    :type aws_access_key_id: string
    :param aws_access_key_id: AWS access key ID
    :type aws_secret_access_key: string
    :param aws_secret_access_key: AWS secret access key
    :type aws_session_token: string
    :param aws_session_token: AWS temporary session token
    :type region_name: string
    :param region_name: Default region when creating new connections
    :type botocore_session: botocore.session.Session
    :param botocore_session: Use this Botocore session instead of creating
                             a new default one.
    :type profile_name: string
    :param profile_name: The name of a profile to use. If not given, then
                         the default profile is used.
    :type aws_account_id: string
    :param aws_account_id: AWS account ID
    """

    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None, region_name=None, botocore_session=None, profile_name=None, aws_account_id=None):
        if botocore_session is not None:
            self._session = botocore_session
        else:
            self._session = botocore.session.get_session()
        if self._session.user_agent_name == 'Botocore':
            botocore_info = f'Botocore/{self._session.user_agent_version}'
            if self._session.user_agent_extra:
                self._session.user_agent_extra += ' ' + botocore_info
            else:
                self._session.user_agent_extra = botocore_info
            self._session.user_agent_name = 'Boto3'
            self._session.user_agent_version = boto3.__version__
        if profile_name is not None:
            self._session.set_config_variable('profile', profile_name)
        credentials_kwargs = {'aws_access_key_id': aws_access_key_id, 'aws_secret_access_key': aws_secret_access_key, 'aws_session_token': aws_session_token, 'aws_account_id': aws_account_id}
        if any(credentials_kwargs.values()):
            if self._account_id_set_without_credentials(**credentials_kwargs):
                raise NoCredentialsError()
            if aws_account_id is None:
                del credentials_kwargs['aws_account_id']
            self._session.set_credentials(*credentials_kwargs.values())
        if region_name is not None:
            self._session.set_config_variable('region', region_name)
        self.resource_factory = ResourceFactory(self._session.get_component('event_emitter'))
        self._setup_loader()
        self._register_default_handlers()

    @property
    def events(self):
        """
        The event emitter for a session
        """
        return self._session.get_component('event_emitter')

    def _setup_loader(self):
        """
        Setup loader paths so that we can load resources.
        """
        self._loader = self._session.get_component('data_loader')
        self._loader.search_paths.append(os.path.join(os.path.dirname(__file__), 'data'))

    def get_partition_for_region(self, region_name):
        """Lists the partition name of a particular region.

        :type region_name: string
        :param region_name: Name of the region to list partition for (e.g.,
             us-east-1).

        :rtype: string
        :return: Returns the respective partition name (e.g., aws).
        """
        return self._session.get_partition_for_region(region_name)

    def _register_default_handlers(self):
        self._session.register('creating-client-class.s3', boto3.utils.lazy_call('boto3.s3.inject.inject_s3_transfer_methods'))
        self._session.register('creating-resource-class.s3.Bucket', boto3.utils.lazy_call('boto3.s3.inject.inject_bucket_methods'))
        self._session.register('creating-resource-class.s3.Object', boto3.utils.lazy_call('boto3.s3.inject.inject_object_methods'))
        self._session.register('creating-resource-class.s3.ObjectSummary', boto3.utils.lazy_call('boto3.s3.inject.inject_object_summary_methods'))
        self._session.register('creating-resource-class.dynamodb', boto3.utils.lazy_call('boto3.dynamodb.transform.register_high_level_interface'), unique_id='high-level-dynamodb')
        self._session.register('creating-resource-class.dynamodb.Table', boto3.utils.lazy_call('boto3.dynamodb.table.register_table_methods'), unique_id='high-level-dynamodb-table')
        self._session.register('creating-resource-class.ec2.ServiceResource', boto3.utils.lazy_call('boto3.ec2.createtags.inject_create_tags'))
        self._session.register('creating-resource-class.ec2.Instance', boto3.utils.lazy_call('boto3.ec2.deletetags.inject_delete_tags', event_emitter=self.events))

    def _account_id_set_without_credentials(self, *, aws_account_id, aws_access_key_id, aws_secret_access_key, **kwargs):
        if aws_account_id is None:
            return False
        elif aws_access_key_id is None or aws_secret_access_key is None:
            return True
        return False
