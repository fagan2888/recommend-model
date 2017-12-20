<?php

use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateTableMfSkfdParas extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('mf_skfd_paras', function($table) {
	    $table->increments('id');
	    $table->string('mf_id')->comment('多因子策略id,0*A股,1*A股基金,2*债券,3*债券基金,4*港股,5*港股基金,6*美股,7*美股基金');
	    $table->string('mf_skid')->comment('对应多因子A股策略id');
	    $table->string('alphakind')->comment('合法性筛选所用的业绩指标');
	    $table->decimal('holdlimit',3,1)->comment('机构持仓比例限制，去掉机构持仓比例在所限制标准差之下的部分，限制标准差数值区分正负');
	    $table->decimal('sizelimit',3,1)->comment('基金规模限制，去掉规模在所限制标准差之上的部分，限制标准差数值区分正负');
	    $table->decimal('alphalimit',3,1)->comment('基金历史业绩限制，去掉业绩在所限制标准差之下的部分，限制标准差数值区分正负');
	    $table->integer('poolsize')->unsigned()->comment('单因子单期基金池规模');
	    $table->integer('sparepoolsize')->unsigned()->comment('备选基金池，即二级池规模，如已选基金未脱出二级池则会保留');
	    $table->date('start_time')->default('0000-00-00')->comment('策略开始时间');
	    $table->timestamps();
	});
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::drop('mf_skfd_paras');
    }
}
