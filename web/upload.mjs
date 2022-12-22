import multer from "multer"
import multerS3 from "multer-s3"
import aws from "aws-sdk"

aws.config.loadFromPath('./config/s3.json');

const s3 = new aws.S3(

);

const uploadMiddleware = multer({
  storage: multerS3({
    s3: s3,
    bucket: 'hoyeon-1',
    acl: 'public-read',
    contentType: multerS3.AUTO_CONTENT_TYPE,
    key: function (req, file, cb) {
      cb(null, `inference/${req.query.tag}/${file.originalname}`);
    },
  }),
});

export default uploadMiddleware