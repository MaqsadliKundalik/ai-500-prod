"""
Seed AI Enhancements Data
==========================
Populates database with sample data for testing AI enhancements
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.models.pharmacy import Pharmacy
from app.models.medication import Medication


async def seed_pharmacy_inventory(session: AsyncSession):
    """Seed pharmacy inventory with price data."""
    print("📦 Seeding pharmacy inventory...")
    
    # Get some pharmacies and medications
    result = await session.execute(text("SELECT id FROM pharmacies LIMIT 10"))
    pharmacies = result.fetchall()
    
    result = await session.execute(text("SELECT id FROM medications LIMIT 20"))
    medications = result.fetchall()
    
    if not pharmacies or not medications:
        print("⚠️  No pharmacies or medications found. Run seed_data.py first!")
        return
    
    inventory_data = []
    for pharmacy in pharmacies:
        pharmacy_id = pharmacy[0]
        # Each pharmacy has 10-15 random medications
        import random
        num_meds = random.randint(10, 15)
        selected_meds = random.sample(medications, num_meds)
        
        for med in selected_meds:
            med_id = med[0]
            base_price = random.randint(5000, 80000)  # UZS
            price_variation = random.uniform(0.8, 1.2)
            price = int(base_price * price_variation)
            
            inventory_data.append({
                'id': str(uuid.uuid4()),
                'pharmacy_id': pharmacy_id,
                'medication_id': med_id,
                'price': price,
                'currency': 'UZS',
                'in_stock': random.random() > 0.2,  # 80% in stock
                'stock_quantity': random.randint(5, 100) if random.random() > 0.2 else 0,
                'last_updated': datetime.utcnow(),
                'created_at': datetime.utcnow()
            })
    
    # Bulk insert
    if inventory_data:
        await session.execute(
            text("""
                INSERT INTO pharmacy_inventory 
                (id, pharmacy_id, medication_id, price, currency, in_stock, stock_quantity, last_updated, created_at)
                VALUES (:id, :pharmacy_id, :medication_id, :price, :currency, :in_stock, :stock_quantity, :last_updated, :created_at)
            """),
            inventory_data
        )
        await session.commit()
        print(f"✅ Seeded {len(inventory_data)} pharmacy inventory records")


async def seed_medication_recalls(session: AsyncSession):
    """Seed medication recalls with sample data."""
    print("⚠️  Seeding medication recalls...")
    
    result = await session.execute(text("SELECT id, name FROM medications LIMIT 5"))
    medications = result.fetchall()
    
    if not medications:
        print("⚠️  No medications found")
        return
    
    recall_data = [
        {
            'id': str(uuid.uuid4()),
            'medication_id': medications[0][0],
            'recall_number': 'FDA-2025-001',
            'source': 'FDA',
            'product_description': f'{medications[0][1]} 500mg tablets',
            'reason': 'Potential contamination with foreign substance',
            'classification': 'Class II',
            'severity': 'HIGH',
            'status': 'active',
            'recall_date': (datetime.utcnow() - timedelta(days=30)).date(),
            'batch_number': 'LOT2025-A123',
            'company': 'Generic Pharma Inc.',
            'distribution_pattern': 'Nationwide',
            'action_required': 'Return to pharmacy for replacement',
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        },
        {
            'id': str(uuid.uuid4()),
            'medication_id': medications[1][0],
            'recall_number': 'WHO-2025-015',
            'source': 'WHO',
            'product_description': f'{medications[1][1]} - Counterfeit Warning',
            'reason': 'Counterfeit product detected in Central Asia',
            'classification': 'Class I',
            'severity': 'CRITICAL',
            'status': 'active',
            'recall_date': (datetime.utcnow() - timedelta(days=15)).date(),
            'batch_number': None,
            'company': 'Unknown',
            'distribution_pattern': 'Uzbekistan, Kazakhstan, Kyrgyzstan',
            'action_required': 'Do not use. Return immediately',
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        },
        {
            'id': str(uuid.uuid4()),
            'medication_id': medications[2][0],
            'recall_number': 'UZ-SOG-2025-042',
            'source': 'UZ_MOH',
            'product_description': f'{medications[2][1]} - Quality Issue',
            'reason': 'Does not meet quality standards (Sifat standartiga mos kelmaydi)',
            'classification': 'Class III',
            'severity': 'MODERATE',
            'status': 'resolved',
            'recall_date': (datetime.utcnow() - timedelta(days=60)).date(),
            'batch_number': 'UZ-2025-B456',
            'company': 'Local Uzbek Manufacturer',
            'distribution_pattern': 'Tashkent, Samarkand regions',
            'action_required': 'Ushbu partiyani qaytaring',
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
    ]
    
    await session.execute(
        text("""
            INSERT INTO medication_recalls 
            (id, medication_id, recall_number, source, product_description, reason, 
             classification, severity, status, recall_date, batch_number, company, 
             distribution_pattern, action_required, created_at, updated_at)
            VALUES (:id, :medication_id, :recall_number, :source, :product_description, 
                    :reason, :classification, :severity, :status, :recall_date, :batch_number, 
                    :company, :distribution_pattern, :action_required, :created_at, :updated_at)
        """),
        recall_data
    )
    await session.commit()
    print(f"✅ Seeded {len(recall_data)} medication recalls")


async def seed_pharmacy_reviews(session: AsyncSession):
    """Seed pharmacy reviews."""
    print("⭐ Seeding pharmacy reviews...")
    
    result = await session.execute(text("SELECT id FROM pharmacies LIMIT 10"))
    pharmacies = result.fetchall()
    
    result = await session.execute(text("SELECT id FROM users LIMIT 20"))
    users = result.fetchall()
    
    if not pharmacies or not users:
        print("⚠️  No pharmacies or users found")
        return
    
    import random
    
    comments_uz = [
        "Xizmat a'lo! Tez va sifatli",
        "Yaxshi dorixona, biroz qimmat",
        "Xodimlar juda do'stona",
        "Doim kerakli dorilar bor",
        "Narxlar o'rtacha, lekin sifat yaxshi",
        "Toza va tartibli dorixona",
        "24 soat ishlashi juda qulay",
        "Ba'zi dorilar yo'q edi",
        "Yomon emas, tavsiya qilaman",
        "Juda qimmat, boshqa yerdan arzon"
    ]
    
    reviews_data = []
    for pharmacy in pharmacies:
        pharmacy_id = pharmacy[0]
        num_reviews = random.randint(5, 15)
        
        for _ in range(num_reviews):
            user = random.choice(users)
            rating = random.choices([5, 4, 3, 2, 1], weights=[50, 30, 15, 4, 1])[0]
            
            reviews_data.append({
                'id': str(uuid.uuid4()),
                'pharmacy_id': pharmacy_id,
                'user_id': user[0],
                'rating': rating,
                'comment': random.choice(comments_uz),
                'service_rating': random.randint(3, 5),
                'price_rating': random.randint(2, 5),
                'availability_rating': random.randint(3, 5),
                'cleanliness_rating': random.randint(4, 5),
                'helpful_count': random.randint(0, 20),
                'created_at': datetime.utcnow() - timedelta(days=random.randint(1, 90)),
                'updated_at': datetime.utcnow()
            })
    
    if reviews_data:
        await session.execute(
            text("""
                INSERT INTO pharmacy_reviews 
                (id, pharmacy_id, user_id, rating, comment, service_rating, price_rating, 
                 availability_rating, cleanliness_rating, helpful_count, created_at, updated_at)
                VALUES (:id, :pharmacy_id, :user_id, :rating, :comment, :service_rating, 
                        :price_rating, :availability_rating, :cleanliness_rating, 
                        :helpful_count, :created_at, :updated_at)
            """),
            reviews_data
        )
        await session.commit()
        print(f"✅ Seeded {len(reviews_data)} pharmacy reviews")


async def seed_user_notifications(session: AsyncSession):
    """Seed user notifications."""
    print("🔔 Seeding user notifications...")
    
    result = await session.execute(text("SELECT id FROM users LIMIT 10"))
    users = result.fetchall()
    
    if not users:
        print("⚠️  No users found")
        return
    
    import random
    
    notification_types = [
        ('recall_alert', 'CRITICAL', '⚠️ Dori chaqirib olinmoqda', 'Siz qabul qilayotgan dorida muammo topildi. Darhol to\'xtating!'),
        ('price_drop', 'INFO', '💰 Narx tushdi!', 'Siz kuzatayotgan dorining narxi 15% arzonlashdi'),
        ('interaction_warning', 'HIGH', '🔴 Xavfli o\'zaro ta\'sir', 'Yangi qo\'shilgan dori boshqa dorilar bilan xavfli ta\'sir qilishi mumkin'),
        ('stock_available', 'INFO', '✅ Dori mavjud', 'Qidirayotgan doringiz endi dorixonada mavjud'),
    ]
    
    notifications_data = []
    for user in users:
        user_id = user[0]
        num_notifications = random.randint(2, 8)
        
        for _ in range(num_notifications):
            notif_type, severity, title, message = random.choice(notification_types)
            is_read = random.random() > 0.4  # 60% read
            
            notifications_data.append({
                'id': str(uuid.uuid4()),
                'user_id': user_id,
                'type': notif_type,
                'title': title,
                'message': message,
                'severity': severity,
                'is_read': is_read,
                'action_url': '/medications/123' if notif_type == 'recall_alert' else None,
                'metadata': None,
                'created_at': datetime.utcnow() - timedelta(days=random.randint(0, 30)),
                'read_at': datetime.utcnow() - timedelta(days=random.randint(0, 10)) if is_read else None
            })
    
    if notifications_data:
        await session.execute(
            text("""
                INSERT INTO user_notifications 
                (id, user_id, type, title, message, severity, is_read, action_url, 
                 metadata, created_at, read_at)
                VALUES (:id, :user_id, :type, :title, :message, :severity, :is_read, 
                        :action_url, :metadata, :created_at, :read_at)
            """),
            notifications_data
        )
        await session.commit()
        print(f"✅ Seeded {len(notifications_data)} user notifications")


async def main():
    """Main seeding function."""
    print("\n" + "="*60)
    print("🌱 SEEDING AI ENHANCEMENTS DATA")
    print("="*60 + "\n")
    
    # Create async engine and session
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        try:
            # Check if tables exist
            result = await session.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('pharmacy_inventory', 'medication_recalls', 
                                   'pharmacy_reviews', 'user_notifications')
            """))
            tables = [row[0] for row in result.fetchall()]
            
            if len(tables) < 4:
                print(f"❌ Missing tables! Found: {tables}")
                print("Run: alembic upgrade head")
                return
            
            # Seed data
            await seed_pharmacy_inventory(session)
            await seed_medication_recalls(session)
            await seed_pharmacy_reviews(session)
            await seed_user_notifications(session)
            
            print("\n" + "="*60)
            print("✅ ALL DATA SEEDED SUCCESSFULLY!")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Error seeding data: {e}")
            await session.rollback()
            raise
        finally:
            await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
