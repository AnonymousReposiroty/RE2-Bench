def bucket_download_file(
    self, Key, Filename, ExtraArgs=None, Callback=None, Config=None
):
    """Download an S3 object to a file.

    Usage::

        import boto3
        s3 = boto3.resource('s3')
        s3.Bucket('amzn-s3-demo-bucket').download_file('hello.txt', '/tmp/hello.txt')

    Similar behavior as S3Transfer's download_file() method,
    except that parameters are capitalized. Detailed examples can be found at
    :ref:`S3Transfer's Usage <ref_s3transfer_usage>`.

    :type Key: str
    :param Key: The name of the key to download from.

    :type Filename: str
    :param Filename: The path to the file to download to.

    :type ExtraArgs: dict
    :param ExtraArgs: Extra arguments that may be passed to the
        client operation. For allowed download arguments see
        :py:attr:`boto3.s3.transfer.S3Transfer.ALLOWED_DOWNLOAD_ARGS`.

    :type Callback: function
    :param Callback: A method which takes a number of bytes transferred to
        be periodically called during the download.

    :type Config: boto3.s3.transfer.TransferConfig
    :param Config: The transfer configuration to be used when performing the
        transfer.
    """
    return self.meta.client.download_file(
        Bucket=self.name,
        Key=Key,
        Filename=Filename,
        ExtraArgs=ExtraArgs,
        Callback=Callback,
        Config=Config,
    )
