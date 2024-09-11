from sqlalchemy import create_engine,Integer, String, DateTime, Column,BigInteger,Date
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# 数据库连接

engine = create_engine('mysql+pymysql://p_HB266:CqLoEPJrAdp4J3b@mdb.servers.dev.ofc:3306/rec')

# Model基类,创建了一个BaseModel类,这个类的子类可以自动与一个表关联
Base = declarative_base()


class ProfileProductBuy(Base):
    '''
    对应数据库中的表
    '''
    __tablename__ = 'ProfileProductBuy'

    id = Column(BigInteger, primary_key=True, nullable=False,autoincrement=True)
    memberId = Column(Integer, nullable=False)
    productCode = Column(String(20), nullable=False)
    thirdCategoryId = Column(BigInteger, nullable=False)
    buyCount = Column(Integer, nullable=False)
    purchaseNum = Column(Integer, nullable=False)
    lastTimePaid = Column(DateTime, nullable=False)
    eventDate = Column(Date, nullable=False)
    
    timeCreated = Column(DateTime, nullable=False, default=datetime.now())
    timeModified = Column(DateTime, nullable=False, default=datetime.now())
    def __repr__(self):
        return "id: {} memberId: {}".format(self.id,self.memberId)

    def __repr__(self):
        return "id: {} memberId: {}".format(self.id,self.memberId)
    
Session = sessionmaker(bind=engine)
session = Session()


profileProductBuys = session.query(ProfileProductBuy).limit(10).all()
print(profileProductBuys)