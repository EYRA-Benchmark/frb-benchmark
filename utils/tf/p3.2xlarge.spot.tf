provider "aws" {
  region = "eu-central-1"
}

resource "aws_key_pair" "deployer" {
  key_name   = "sshkey"
  public_key = "${file("id_rsa.pub")}"
}

resource "aws_security_group" "sec" {
  name = "sec"
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_spot_instance_request" "liam" {
  instance_type = "p3.2xlarge"
  spot_price    = "1.35"
  availability_zone        = "eu-central-1b"
  vpc_security_group_ids = ["${aws_security_group.sec.id}"]
  ami = data.aws_ami.ubuntu.id # ubuntu 18
  key_name = "sshkey"
  root_block_device {
    volume_size = "15" #GB
    #delete_on_termination = false
  }

}
data "aws_ami" "ubuntu" {
    most_recent = true

    filter {
        name   = "name"
        values = ["ubuntu/images/hvm-ssd/ubuntu-bionic-18.04-amd64-server-*"]
    }

    filter {
        name   = "virtualization-type"
        values = ["hvm"]
    }

    owners = ["099720109477"] # Canonical
}

output "image_id" {
    value = data.aws_ami.ubuntu.id
}
